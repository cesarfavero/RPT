# RPT

Версии README:

- English: [`README.md`](/Users/cesarfavero/Documents/MyPru/RPT/README.md)
- Português: [`README.pt-BR.md`](/Users/cesarfavero/Documents/MyPru/RPT/README.pt-BR.md)
- 中文: [`README.zh-CN.md`](/Users/cesarfavero/Documents/MyPru/RPT/README.zh-CN.md)
- Русский: `README.ru.md`

Это исследовательский и инженерный репозиторий с двумя связанными направлениями:

1. исследование `BitNet / sparsity / QAT-STE`, которое на раннем этапе помогло paper-треку и методологии;
2. `gemma4_theory/` — текущий прикладной трек: взять современную плотную Gemma 2B, выполнить pruning, тернаризовать MLP-путь, восстановить функциональность и собрать пригодный bundle с Triton runtime на основе нашего собственного метода и runtime-стека.

## Намерение

Этот репозиторий является исследовательским и инженерным проектом.

Он фиксирует, что было протестировано, что сработало, что не сработало и что на данный момент уже пригодно к использованию. Это не коммерческая презентация; все утверждения здесь следует понимать как технические утверждения, ограниченные реально проверенными benchmark'ами и средами.

## Что Сейчас Важнее Всего

Если вам нужен текущий практический результат, начните с [`gemma4_theory/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/README.md).

В этом треке уже подтверждено на локальном benchmark'е из 100 вопросов:

- исходный HF `google/gemma-2-2b-it`: `0.90`
- наш улучшенный плотный checkpoint `student_pruned`: `0.92`
- наш packed ternary путь после recovered adapter: `1.00`
- наш интегрированный Triton runtime для packed пути: `1.00`

Это самый важный текущий вывод репозитория:

- мы улучшили локальный результат по сравнению с исходным Gemma 2B;
- мы перевели весь MLP-путь в packed ternary представление;
- мы восстановили функциональность поверх этого packed-пути;
- мы собрали рабочий runtime bundle для Linux/NVIDIA/Colab.

## От Microsoft Как Ориентира К Нашему Собственному Стеку

Материалы BitNet/Microsoft были полезны в начале как концептуальный ориентир для тернаризации, pruning и дисциплины деплоя.

Но текущий Gemma-трек — это не копия Microsoft BitNet runtime. Практический стек в этом репозитории теперь наш собственный:

- наш собственный путь pruning/recovery поверх Gemma 2B
- наш собственный экспорт packed ternary MLP
- наш собственный процесс packed-path recovery
- наш собственный structural runtime и Triton runtime путь
- наш собственный production bundle для Linux/NVIDIA/Colab

Честная формулировка такая:

- Microsoft/BitNet был полезной отправной точкой
- текущий рабочий метод и runtime — наши

## Текущее Состояние По Направлениям

### 1. BitNet / Paper Track

В репозитории по-прежнему сохранена более ранняя работа, ориентированная на BitNet:

- исходник статьи: `paper/rpt_sparsity_paper.tex`
- скрипты деплоя: `deploy/`
- исторические notebooks и логи

Это по-прежнему важно для исследовательской истории, но уже не является единственной активной ценностью репозитория.

### 2. Gemma Track

Текущий практический путь находится в `gemma4_theory/`.

Ключевые подтвержденные этапы:

- исходный HF Gemma 2B на локальном наборе: `0.90`
- наш улучшенный плотный checkpoint `student_pruned`: `0.92`
- полное восстановление после полной MLP ternary expansion: `0.92`
- recovered packed/materialized path: `1.00`
- интегрированный Triton runtime: `1.00`
- проверка production bundle после prewarm на T4:
  - `avg_tokens_per_sec ~= 5.66`
  - `min_latency ~= 0.35s`
  - `max_latency ~= 1.02s`

Это означает, что проект уже вышел за рамки «только теории». Сейчас у него уже есть рабочий runtime bundle для поддерживаемой среды, и текущий практический путь больше не зависит от Microsoft BitNet tooling.

## Отдельный Репозиторий Runtime

Отдельная работа по runtime и CLI теперь вынесена в отдельный репозиторий:

- [cesarfavero/rpt-runtime](https://github.com/cesarfavero/rpt-runtime)

Основной репозиторий RPT остается источником истины для исследования и валидации.

## С Чего Начать

### Если вам нужен исследовательский нарратив

Откройте:

- [`gemma4_theory/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/README.md)

### Если вам нужен packed artifact и runtime

Откройте:

- [`gemma4_theory/deploy/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/deploy/README.md)
- [`gemma4_theory/runtime/packed_recovered_l78/README.md`](/Users/cesarfavero/Documents/MyPru/RPT/gemma4_theory/runtime/packed_recovered_l78/README.md)

### Если вы уже скачали production zip

Рекомендуемый поток:

```bash
unzip production_triton_packed_recovered_l78.zip
cd production_triton_packed_recovered_l78
python -m venv .venv
source .venv/bin/activate
pip install -r runtime/requirements.txt
python runtime/run_triton_inference.py --model-dir . --prompt "Qual a capital da França?"
```

Подробности описаны в README для runtime и deploy.

## Структура Репозитория

- `paper/`: исходники статьи и файлы, связанные с публикацией.
- `deploy/`: legacy BitNet deploy track.
- `gemma4_theory/`: текущий путь dense -> pruned -> ternary -> runtime.
- `sessions/`: исторические журналы экспериментов.
- `results/`: отслеживаемые JSON/CSV результаты.

## Воспроизводимость

Правило репозитория простое:

- notebooks сохраняют timestamped артефакты;
- результаты не редактируются вручную;
- каждый полезный этап должен завершаться commit + push.

## Лицензия

MIT License (`LICENSE`).
