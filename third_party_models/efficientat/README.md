# EfficientAT MN04 AudioSet

Этот каталог содержит локальную модель классификации аудио, используемую
DubClean для подтверждения продолжительных песенных интервалов только на
оригинальной английской дорожке.

- Исходный проект: https://github.com/fschmid56/EfficientAT
- Архитектура: `mn04_as`
- Исходные веса: `mn04_as_mAP_432.pt`
- Публикация весов: https://github.com/fschmid56/EfficientAT/releases/tag/v0.0.1
- Лицензия: MIT, см. `../../licenses/EfficientAT-MIT.txt`
- Выходные классы: 527 классов AudioSet

Файл `efficientat_mn04_audioset_waveform.pt` — локальный TorchScript-экспорт
модели и её штатной log-mel-подготовки. На вход подаётся монофонический звук
32 кГц; сеть не обращается к интернету и не загружает веса во время работы.

DubClean использует оценки `Singing`, `Song`, `Vocal music` и `Music` для
подтверждения песни. Оценка `Speech` сохраняется в отчёте, но не отменяет
песню с наложенным диалогом. Восстановление разрешается лишь после проверки
длительности, музыкального фона и фактического удаления вокала.

Источник скачивания: https://raw.githubusercontent.com/yaromanol1/DubClean/v0.6.0/third_party_models/efficientat/efficientat_mn04_audioset_waveform.pt
