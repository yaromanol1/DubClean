"""Contract tests for the file the user actually walks away with.

The final film used to be a technically valid Matroska that ordinary players
choked on: cover art demoted into extra video tracks, lossless audio no TV or
online service accepts, and no way for a cached build to notice that the
recipe had changed.  These tests pin the corrected contract.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from experiments.paired_reference_cancel import audio_io
from experiments.paired_reference_cancel.application_pipeline import (
    _align_to_dubbed,
    _alignment_coverage,
    _alignment_segment_speed_is_safe,
    _cached_model_output_valid_for,
    _inverted_alignment,
    _record_model_output_for,
    _remux_recipe_signature,
    _source_language,
)


ROOT = Path(__file__).parents[2]


def _pairs(command: list[str], flag: str) -> list[str]:
    return [command[i + 1] for i, item in enumerate(command[:-1]) if item == flag]


class RemuxCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.command = audio_io.build_remux_command(
            "ffmpeg",
            "movie.mkv",
            "clean.flac",
            "out.mkv",
            audio_codec="aac",
            audio_bitrate="256k",
            audio_channels=2,
            audio_sample_rate=48000,
            track_title="Русская дорожка",
        )

    def test_cover_art_never_becomes_a_video_track(self) -> None:
        # Lowercase "v" also matches the attached_pic streams Matroska cover
        # attachments are demuxed as.  Copying those wrote extra video tracks
        # that announce ~30000 fps and stop delivering frames after t=0.
        maps = _pairs(self.command, "-map")

        self.assertIn("0:V:0?", maps)
        self.assertNotIn("0:v", maps)
        self.assertNotIn("0:v?", maps)

    def test_cleaned_track_is_the_first_and_only_default_audio(self) -> None:
        maps = _pairs(self.command, "-map")

        self.assertEqual(
            [item for item in maps if item.startswith("1:")],
            ["1:a:0"],
        )
        self.assertEqual(maps.index("1:a:0"), maps.index("0:V:0?") + 1)
        self.assertNotIn("0:a?", maps)
        self.assertIn("default", _pairs(self.command, "-disposition:a:0"))

    def test_copied_stream_flags_are_overridden_explicitly(self) -> None:
        # Copied streams inherit the source dispositions, so a source subtitle
        # flagged "default" would switch itself on over the cleaned dub.
        self.assertEqual(_pairs(self.command, "-disposition:s"), ["0"])
        self.assertEqual(_pairs(self.command, "-disposition:v:0"), ["default"])

    def test_subtitles_chapters_and_metadata_survive(self) -> None:
        maps = _pairs(self.command, "-map")

        self.assertIn("0:s?", maps)
        self.assertIn("0:t?", maps)
        self.assertIn("-map_chapters", self.command)
        self.assertIn("-map_metadata", self.command)

    def test_audio_format_is_applied_only_to_the_cleaned_track(self) -> None:
        self.assertEqual(_pairs(self.command, "-c"), ["copy"])
        self.assertEqual(_pairs(self.command, "-c:a:0"), ["aac"])
        self.assertEqual(_pairs(self.command, "-b:a:0"), ["256k"])
        self.assertEqual(_pairs(self.command, "-ac:a:0"), ["2"])
        self.assertEqual(_pairs(self.command, "-ar:a:0"), ["48000"])
        self.assertEqual(
            _pairs(self.command, "-metadata:s:a:0"),
            [
                f"title=Русская дорожка · {audio_io.PROCESSED_TRACK_SUFFIX}",
                "language=rus",
            ],
        )

    def test_optional_audio_format_arguments_are_omitted(self) -> None:
        command = audio_io.build_remux_command(
            "ffmpeg", "movie.mkv", "clean.flac", "out.mkv", audio_codec="flac"
        )

        self.assertNotIn("-b:a:0", command)
        self.assertNotIn("-ac:a:0", command)
        self.assertNotIn("-ar:a:0", command)
        self.assertEqual(_pairs(command, "-c:a:0"), ["flac"])

    def test_output_is_the_last_argument(self) -> None:
        self.assertEqual(self.command[-1], "out.mkv")

    def test_cleaned_track_is_labelled_as_processed(self) -> None:
        self.assertIn(
            f"title=Русская дорожка · {audio_io.PROCESSED_TRACK_SUFFIX}",
            _pairs(self.command, "-metadata:s:a:0"),
        )


class AllTracksRemuxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.command = audio_io.build_remux_command(
            "ffmpeg",
            "movie.mkv",
            "clean.flac",
            "out.mkv",
            audio_codec="aac",
            audio_bitrate="256k",
            audio_sample_rate=48000,
            track_title="Русская дорожка",
            extra_audio=[
                {
                    "path": "companion.flac",
                    "title": "Оригинальная дорожка · синхронизирована",
                    "language": "eng",
                }
            ],
            keep_source_audio=True,
        )

    def test_every_extra_file_becomes_its_own_input(self) -> None:
        self.assertEqual(
            _pairs(self.command, "-i"), ["movie.mkv", "clean.flac", "companion.flac"]
        )

    def test_track_order_puts_the_cleaned_track_first(self) -> None:
        maps = _pairs(self.command, "-map")

        self.assertEqual(
            maps, ["0:V:0?", "1:a:0", "2:a:0", "0:a?", "0:s?", "0:t?"]
        )

    def test_source_tracks_lose_their_inherited_default_flag(self) -> None:
        # "-disposition:a 0" must precede the grant, otherwise the source's own
        # default audio track keeps competing with the cleaned one.
        clear = self.command.index("-disposition:a")
        grant = self.command.index("-disposition:a:0")

        self.assertLess(clear, grant)
        self.assertEqual(self.command[clear + 1], "0")
        self.assertEqual(self.command[grant + 1], "default")

    def test_extra_tracks_are_re_encoded_and_labelled(self) -> None:
        self.assertEqual(_pairs(self.command, "-c:a:1"), ["aac"])
        self.assertEqual(_pairs(self.command, "-b:a:1"), ["256k"])
        self.assertEqual(
            _pairs(self.command, "-metadata:s:a:1"),
            ["title=Оригинальная дорожка · синхронизирована", "language=eng"],
        )

    def test_extra_tracks_never_take_the_channel_downmix(self) -> None:
        # Only the cleaned track is forced to stereo; a companion programme
        # keeps whatever layout it was rendered with.
        self.assertNotIn("-ac:a:1", self.command)

    def test_unknown_language_is_declared_rather_than_guessed(self) -> None:
        command = audio_io.build_remux_command(
            "ffmpeg",
            "movie.mkv",
            "clean.flac",
            "out.mkv",
            audio_codec="aac",
            extra_audio=[{"path": "companion.flac", "title": "Оригинал"}],
        )

        self.assertIn("language=und", _pairs(command, "-metadata:s:a:1"))

    def test_clean_only_build_keeps_no_source_audio(self) -> None:
        command = audio_io.build_remux_command(
            "ffmpeg", "movie.mkv", "clean.flac", "out.mkv", audio_codec="aac"
        )

        self.assertNotIn("0:a?", _pairs(command, "-map"))


class ShippedRemuxConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(
            (
                ROOT
                / "python_src"
                / "experiments"
                / "paired_reference_cancel"
                / "config.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_final_film_ships_a_universally_playable_audio_codec(self) -> None:
        full_processing = self.config["full_processing"]

        self.assertEqual(full_processing["remux_audio_codec"], "aac")
        self.assertEqual(full_processing["remux_audio_bitrate"], "256k")
        self.assertEqual(full_processing["remux_audio_channels"], 2)
        self.assertEqual(full_processing["remux_audio_sample_rate"], 48000)


class RemuxCacheInvalidationTests(unittest.TestCase):
    def test_recipe_change_forces_the_final_film_to_be_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dependency = root / "clean.flac"
            dependency.write_bytes(b"audio")
            movie = root / "result.mkv"
            movie.write_bytes(b"movie")
            old = _remux_recipe_signature(
                {"remux_container": "mkv", "remux_audio_codec": "flac"}
            )
            new = _remux_recipe_signature(
                {
                    "remux_container": "mkv",
                    "remux_audio_codec": "aac",
                    "remux_audio_bitrate": "256k",
                }
            )
            _record_model_output_for(movie, [dependency], recipe=old)

            self.assertTrue(
                _cached_model_output_valid_for(movie, [dependency], recipe=old)
            )
            self.assertFalse(
                _cached_model_output_valid_for(movie, [dependency], recipe=new)
            )

    def test_films_built_before_the_recipe_existed_are_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dependency = root / "clean.flac"
            dependency.write_bytes(b"audio")
            movie = root / "result.mkv"
            movie.write_bytes(b"movie")
            _record_model_output_for(movie, [dependency])
            recipe = _remux_recipe_signature({"remux_audio_codec": "aac"})

            self.assertFalse(
                _cached_model_output_valid_for(movie, [dependency], recipe=recipe)
            )

    def test_recipe_free_artifacts_keep_their_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dependency = root / "checkpoint.pt"
            dependency.write_bytes(b"weights")
            artifact = root / "voice.flac"
            artifact.write_bytes(b"audio")
            _record_model_output_for(artifact, [dependency])

            self.assertTrue(_cached_model_output_valid_for(artifact, [dependency]))
            sidecar = artifact.with_name(artifact.name + ".source.json")
            self.assertNotIn(
                "recipe", json.loads(sidecar.read_text(encoding="utf-8"))
            )

    def test_both_pictures_ship_the_other_side_as_a_synced_track(self) -> None:
        source = (
            ROOT
            / "python_src"
            / "experiments"
            / "paired_reference_cancel"
            / "application_pipeline.py"
        ).read_text(encoding="utf-8")
        block = source.split('if bool(parameters.get("remux", True)):', 1)[1]

        # Dubbed picture: the original programme is already on this timeline.
        self.assertIn('"title": "Оригинальная дорожка · синхронизирована",', block)
        # Original picture: the translation has to be brought over to it, or
        # choosing the better picture silently drops the untouched dub.
        self.assertIn('"title": "Дорожка перевода · синхронизирована",', block)
        self.assertIn("dubbed_program_original_timeline.flac", block)
        self.assertIn(
            "_source_language(pair, \"dubbed\")", block
        )

    def test_retimed_companion_is_never_taken_from_the_raw_other_file(self) -> None:
        source = (
            ROOT
            / "python_src"
            / "experiments"
            / "paired_reference_cancel"
            / "application_pipeline.py"
        ).read_text(encoding="utf-8")
        block = source.split('if bool(parameters.get("remux", True)):', 1)[1]

        # Only files the pipeline placed on the picture's timeline may join.
        # Handing ffmpeg the other source container directly would ship tracks
        # that drift by the whole speed difference between the two releases.
        self.assertNotIn('"path": str(dubbed_video)', block)
        self.assertNotIn('"path": str(original_video)', block)

    def test_recipe_signature_tracks_every_format_choice(self) -> None:
        base = {
            "remux_container": "mkv",
            "remux_audio_codec": "aac",
            "remux_audio_bitrate": "256k",
            "remux_audio_channels": 2,
            "remux_audio_sample_rate": 48000,
        }
        signature = _remux_recipe_signature(base)

        self.assertIn(audio_io.REMUX_RECIPE_VERSION, signature)
        for key, changed in [
            ("remux_container", "mp4"),
            ("remux_audio_codec", "ac3"),
            ("remux_audio_bitrate", "448k"),
            ("remux_audio_channels", 6),
            ("remux_audio_sample_rate", 44100),
        ]:
            with self.subTest(key=key):
                self.assertNotEqual(
                    signature, _remux_recipe_signature({**base, key: changed})
                )


class InverseAlignmentTests(unittest.TestCase):
    """Swapping the picture to the original requires an exact inverse map."""

    def setUp(self) -> None:
        # PAL speed-up: the original runs 25 fps, the dub 23.976, so the dubbed
        # timeline is ~4% longer and a whole-film constant scale applies.
        self.alignment = {
            "schema_version": 2,
            "method": "first_match_sparse_verification_v3",
            "summary": {"processing_max_speed_deviation": 0.06},
            "manual_correction_sec": 0.0,
            "segments": [
                {
                    "dubbed_start": 0.0,
                    "dubbed_end": 8.0,
                    "original_start": 26.25,
                    "original_end": 34.25,
                    "duration": 8.0,
                    "speed_ratio": 1.0,
                    "usable": False,
                },
                {
                    "dubbed_start": 8.0,
                    "dubbed_end": 5740.16,
                    "original_start": 34.25,
                    "original_end": 5531.65,
                    "duration": 5732.16,
                    "speed_ratio": 0.95904068,
                    "usable": True,
                },
            ],
        }

    @staticmethod
    def _forward(alignment: dict, moment: float) -> float | None:
        correction = float(alignment.get("manual_correction_sec") or 0.0)
        for item in alignment["segments"]:
            if item.get("usable") is False:
                continue
            if float(item["dubbed_start"]) <= moment < float(item["dubbed_end"]):
                return (
                    float(item["original_start"])
                    + (moment - float(item["dubbed_start"]))
                    * float(item["speed_ratio"])
                    + correction
                )
        return None

    def test_round_trip_returns_the_same_moment(self) -> None:
        inverted = _inverted_alignment(self.alignment)

        for moment in (100.0, 1500.0, 3000.0, 5000.0, 5700.0):
            with self.subTest(moment=moment):
                on_original = self._forward(self.alignment, moment)
                self.assertIsNotNone(on_original)
                self.assertAlmostEqual(
                    self._forward(inverted, on_original), moment, places=6
                )

    def test_inverted_rate_is_reciprocal_and_still_within_the_safety_limit(
        self,
    ) -> None:
        inverted = _inverted_alignment(self.alignment)
        moving = inverted["segments"][-1]

        self.assertAlmostEqual(
            float(moving["speed_ratio"]), 1.0 / 0.95904068, places=9
        )
        self.assertTrue(_alignment_segment_speed_is_safe(inverted, moving))

    def test_manual_correction_is_folded_into_the_output_span(self) -> None:
        corrected = {**self.alignment, "manual_correction_sec": 1.5}

        inverted = _inverted_alignment(corrected)

        self.assertEqual(inverted["manual_correction_sec"], 0.0)
        self.assertAlmostEqual(
            float(inverted["segments"][-1]["dubbed_start"]), 34.25 + 1.5, places=6
        )
        for moment in (600.0, 4200.0):
            with self.subTest(moment=moment):
                on_original = self._forward(corrected, moment)
                self.assertAlmostEqual(
                    self._forward(inverted, on_original), moment, places=6
                )

    def test_unusable_segments_survive_but_never_count_as_coverage(self) -> None:
        inverted = _inverted_alignment(self.alignment)

        self.assertEqual(len(inverted["segments"]), 2)
        self.assertFalse(inverted["segments"][0]["usable"])
        coverage = _alignment_coverage(inverted, 5531.65)
        self.assertGreater(coverage, 0.98)
        self.assertLessEqual(coverage, 1.0)

    def test_degenerate_segments_are_dropped(self) -> None:
        broken = {
            **self.alignment,
            "segments": [
                {
                    "dubbed_start": 10.0,
                    "dubbed_end": 5.0,
                    "original_start": 0.0,
                    "speed_ratio": 1.0,
                },
                {
                    "dubbed_start": 0.0,
                    "dubbed_end": 10.0,
                    "original_start": 0.0,
                    "speed_ratio": 0.0,
                },
                {
                    "dubbed_start": 0.0,
                    "dubbed_end": 10.0,
                    "original_start": 0.0,
                    "speed_ratio": float("nan"),
                },
            ],
        }

        self.assertEqual(_inverted_alignment(broken)["segments"], [])

    def test_round_trip_through_the_real_writer_recovers_the_source(self) -> None:
        """Forward then inverse must land back on the same moments.

        The arithmetic is checked above; this drives the actual writer, so a
        mistake in segment stitching or resampling cannot slip through.
        """
        rate = 24000
        original_duration = 90.0
        offset = 12.0
        speed = 0.95904068
        dubbed_span = (original_duration - offset) / speed
        alignment = {
            "summary": {"processing_max_speed_deviation": 0.06},
            "manual_correction_sec": 0.0,
            "segments": [
                {
                    "dubbed_start": 0.0,
                    "dubbed_end": 4.0,
                    "original_start": 0.0,
                    "original_end": 4.0,
                    "speed_ratio": 1.0,
                    "usable": False,
                },
                {
                    "dubbed_start": 4.0,
                    "dubbed_end": 4.0 + dubbed_span,
                    "original_start": offset,
                    "original_end": original_duration,
                    "speed_ratio": speed,
                    "usable": True,
                },
            ],
        }
        moments = np.arange(20.0, original_duration - 5.0, 10.0)
        samples = np.arange(int(original_duration * rate)) / rate
        body = 0.25 * np.sin(2.0 * np.pi * 180.0 * samples)
        for moment in moments:
            start = int(moment * rate)
            body[start : start + 120] = 0.9

        class _Silent:
            def update(self, **_kwargs: object) -> None:
                pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "original.flac"
            sf.write(str(source), body.astype(np.float32), rate, format="FLAC")
            forward = root / "on_dubbed.flac"
            _align_to_dubbed(
                source, forward, alignment, rate,
                4.0 + dubbed_span, 1, _Silent(), "t", "forward",
            )
            back = root / "back.flac"
            _align_to_dubbed(
                back.parent / forward.name, back, _inverted_alignment(alignment),
                rate, original_duration, 1, _Silent(), "t", "inverse",
            )
            restored, _ = sf.read(str(back), dtype="float32")

        for moment in moments:
            with self.subTest(moment=moment):
                centre = int(moment * rate)
                window = restored[centre - 60 : centre + 180]
                peak = int(np.argmax(np.abs(window))) - 60
                # The marker must come back within a millisecond of where it
                # started, otherwise the film's sound drifts from its picture.
                self.assertLess(abs(peak) / rate, 0.001)

    def test_coverage_of_a_map_full_of_holes_stays_low(self) -> None:
        sparse = {
            **self.alignment,
            "segments": [
                {
                    "dubbed_start": 0.0,
                    "dubbed_end": 500.0,
                    "original_start": 0.0,
                    "duration": 500.0,
                    "speed_ratio": 1.0,
                    "usable": True,
                }
            ],
        }

        self.assertLess(_alignment_coverage(sparse, 5531.65), 0.1)


class SourceLanguageTests(unittest.TestCase):
    def test_language_comes_from_the_selected_stream(self) -> None:
        pair = {
            "sources": {
                "original": {
                    "stream_index": 2,
                    "probe": {
                        "streams": [
                            {"index": 1, "language": "rus"},
                            {"index": 2, "language": "eng"},
                        ]
                    },
                }
            }
        }

        self.assertEqual(_source_language(pair, "original"), "eng")

    def test_missing_language_is_reported_as_undetermined(self) -> None:
        pair = {
            "sources": {
                "original": {
                    "stream_index": 2,
                    "probe": {"streams": [{"index": 2, "language": None}]},
                }
            }
        }

        self.assertEqual(_source_language(pair, "original"), "und")

    def test_absent_source_never_raises(self) -> None:
        self.assertEqual(_source_language({}, "original"), "und")


class ResultFolderRevealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html_source = (
            ROOT / "docs" / "концепт интерфейса" / "DubClean-RU.dc.html"
        ).read_text(encoding="utf-8")

    def test_reveal_targets_the_assembled_film_before_any_track(self) -> None:
        reveal = self.html_source.split("async revealResultFolder() {", 1)[1].split(
            "\n  }", 1
        )[0]

        # The published FLAC lives in the "Дорожки DubClean" subfolder, which
        # does not contain result.mkv at all.
        self.assertIn(
            "const path = result.movie || result.user_files?.audio "
            "|| result.audio || result.manifest;",
            reveal,
        )

    def test_video_choice_appears_only_for_two_distinct_files(self) -> None:
        # A single container holding both tracks has one picture, and a source
        # without a video stream cannot supply one either.
        self.assertIn(
            "originalPath.toLowerCase() !== dubbedPath.toLowerCase()",
            self.html_source,
        )
        self.assertIn("&& !!sources.original?.probe?.video;", self.html_source)

    def test_video_choice_visibility_has_one_definition(self) -> None:
        # Visibility and the resolved source are decided together inside
        # packagingChoice; a second copy in the renderer could disagree with it.
        self.assertEqual(
            self.html_source.count("const videoChoiceVisible = !!originalPath"), 1
        )
        self.assertIn(
            "const videoChoiceVisible = packaging.videoChoiceVisible;",
            self.html_source,
        )

    def test_packaging_choices_reach_the_backend(self) -> None:
        run_full = self.html_source.split("async runBackendFull() {", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertIn("video_source:videoSource", run_full)
        self.assertIn("include_all_tracks:includeAllTracks", run_full)

    def test_per_project_choice_starts_unset_so_defaults_apply(self) -> None:
        # A concrete initial value would silently ignore whatever the operator
        # set in Settings; empty and null mean "ask the defaults".
        self.assertIn("defaultVideoSource: 'auto',", self.html_source)
        self.assertIn("defaultIncludeAllTracks: true,", self.html_source)
        self.assertIn("videoSource: '',", self.html_source)
        self.assertIn("includeAllTracks: null,", self.html_source)

    def test_defaults_are_offered_in_settings(self) -> None:
        self.assertIn("label:'Итоговый фильм: видео',", self.html_source)
        self.assertIn("label:'Итоговый фильм: дорожки',", self.html_source)
        self.assertIn(
            "(i)=>this.setPreference({ defaultVideoSource:"
            "['auto','dubbed','original'][i] })",
            self.html_source,
        )
        self.assertIn(
            "(i)=>this.setPreference({ defaultIncludeAllTracks: i === 0 })",
            self.html_source,
        )

    def test_defaults_survive_a_restart(self) -> None:
        self.assertIn(
            "if (['auto','dubbed','original'].includes(saved.defaultVideoSource))",
            self.html_source,
        )
        self.assertIn(
            "if (typeof saved.defaultIncludeAllTracks === 'boolean')",
            self.html_source,
        )
        self.assertIn("defaultVideoSource: state.defaultVideoSource,", self.html_source)
        self.assertIn(
            "defaultIncludeAllTracks: state.defaultIncludeAllTracks,",
            self.html_source,
        )

    def test_best_quality_is_decided_by_resolution_then_bitrate(self) -> None:
        method = self.html_source.split("  packagingChoice() {", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertIn("const originalPixels = Number(originalVideo.pixels || 0);", method)
        self.assertIn("const dubbedPixels = Number(dubbedVideo.pixels || 0);", method)
        self.assertIn("originalPixels !== dubbedPixels", method)
        # A tie must fall to the dubbed picture: it needs no retiming, so the
        # build cannot later be refused over inverse-map coverage.
        self.assertIn(
            "? 'original' : 'dubbed')", method
        )
        self.assertIn("Number(originalVideo.bit_rate || 0)", method)

    def test_screen_and_build_read_the_same_choice(self) -> None:
        # One definition, three callers: the preview screen, the step-by-step
        # build and the one-click build. A second copy of the rule would let
        # the screen show one thing while the assembled film used another.
        self.assertEqual(self.html_source.count("this.packagingChoice()"), 3)
        run_full = self.html_source.split("async runBackendFull() {", 1)[1].split(
            "\n  }", 1
        )[0]
        self.assertIn(
            "const { videoSource, includeAllTracks } = this.packagingChoice();",
            run_full,
        )

    def test_hidden_choice_can_never_send_the_original(self) -> None:
        method = self.html_source.split("  packagingChoice() {", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertIn(
            "videoSource: videoChoiceVisible && requested === 'original' "
            "? 'original' : 'dubbed',",
            method,
        )

    def test_packaging_choices_live_on_the_preview_screen(self) -> None:
        # They are run parameters, so they belong where the run is configured,
        # not on the assembly screen where the choice has already been made.
        preview = self.html_source.split("<!-- SCREEN 5: PREVIEW -->", 1)[1].split(
            "<!-- SCREEN 6:", 1
        )[0]
        export = self.html_source.split("<!-- SCREEN 7:", 1)[1].split(
            "<!-- SCREEN 8:", 1
        )[0]

        self.assertIn("{{ videoSourceOptions }}", preview)
        self.assertIn("{{ trackSetOptions }}", preview)
        self.assertNotIn("videoSourceOptions", export)
        self.assertNotIn("trackSetOptions", export)

    def test_packaging_explanations_live_in_the_help_dot(self) -> None:
        # Prose belongs behind the question mark next to a heading, the pattern
        # every other screen uses; it must not sit on the screen as body text.
        packaging = self.html_source.split(
            '<div style="font-weight:700;font-size:15px">Итоговый фильм</div>', 1
        )[1].split('<sc-if value="{{ previewNotReady }}"', 1)[0]

        # One for the card itself plus one per option group.
        self.assertEqual(packaging.count('dc-import name="HelpDot"'), 3)
        for phrase in ("обратной временной карте", "другом таймлайне"):
            with self.subTest(phrase=phrase):
                position = packaging.find(phrase)
                self.assertGreater(position, 0)
                # Every explanatory phrase must sit inside a HelpDot's text
                # attribute, never in a rendered element.
                self.assertIn(
                    "HelpDot", packaging[max(0, position - 400):position]
                )

    def test_reveal_reads_the_same_sources_as_the_enabled_state(self) -> None:
        reveal = self.html_source.split("async revealResultFolder() {", 1)[1].split(
            "\n  }", 1
        )[0]

        self.assertIn("pair.application_result || {}", reveal)
        self.assertIn("pair.application_full_progress || {}", reveal)
        self.assertIn("openResultDisabled: !resultReady", self.html_source)


if __name__ == "__main__":
    unittest.main()
