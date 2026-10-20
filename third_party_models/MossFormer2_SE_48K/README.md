# MossFormer Plus — дообученная модель DubClean

`last_best_checkpoint.pt` — производный чекпойнт MossFormer2 SE 48K,
дообученный для извлечения общей речи EN+RU из дорожки перевода. Это не
исходный checkpoint, опубликованный командой ClearerVoice-Studio.

- базовая модель и код: <https://github.com/modelscope/ClearerVoice-Studio>;
- базовая лицензия: Apache License 2.0;
- лицензия производного чекпойнта: Apache License 2.0;
- ожидаемый размер: 221 528 916 байт;
- SHA-256: `01D6003C19FFF3B7597EB275AFB65C56554E4C2D4A31EDF0001E2FF9229035BC`;
- обучающие материалы в репозиторий и релизный комплект не входят.

Публичный экспорт содержит только параметры сети: все тензоры совпадают
с полным авторским чекпойнтом. Оптимизатор и история обучения исключены.
Файл сохранён в GitHub Releases и автоматически устанавливается через SETUP.bat.

Источник скачивания: https://github.com/yaromanol1/DubClean/releases/download/v0.6.0/mossformer-plus-48k.pt
