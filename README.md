# Стенд для сравнения методов распределения задач в группе беспилотных аппаратов

Программный код и данные к статье «Выбор метода распределения задач в группе беспилотных аппаратов
в зависимости от скученности точек и критерия качества» (выходные данные будут добавлены после
публикации).

Группа из M аппаратов должна посетить N точек. Стенд сравнивает пять методов распределения
точек между аппаратами с фронтом Парето по суммарному пути и времени выполнения: при N ≤ 24 фронт
находится точно (CP-SAT), при большем N эталоном служит лучшее известное решение. Главный фактор
эксперимента — скученность точек, от равномерного поля до сильных скоплений. Всего рассчитано
12 500 тестовых примеров.

*English summary is at the end.*

## Воспроизвести таблицы и рисунки статьи

Нужен Python 3.11–3.13. Около 5 минут.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m experiments.article
```

На Windows то же самое делает двойной щелчок по `scripts\run_article_windows.bat`.

Результат появляется в папке `results/article/`:

- `table1_reference.csv` … `table5_gaca.csv` — таблицы 1–5;
- `fig1_fields.png` … `fig4_bestmap.png` — рисунки 1–4;
- `report.md` и `numbers.csv` — сверка: каждое число из текста статьи рядом с пересчитанным
  значением и отметкой о совпадении.

Таблицы и рисунки строятся из готовых данных в папке `data/`, кроме табл. 3: для неё методы заново
прогоняются на тех же примерах. Описание файлов и столбцов — в [`data/README.md`](data/README.md).

## Что в репозитории

```
allocsim/               библиотека
  fields.py             поля задач: равномерные и со скоплениями (процесс Томаса)
  clustering.py         показатель скученности R (Кларк — Эванс)
  instance.py           тестовый пример: точки, типы задач, умения аппаратов, старты
  plan.py               план эксперимента: ячейки основного эксперимента и проверок
  exact.py              точное решение и фронт Парето (CP-SAT)
  best_known.py         лучшее известное решение (маршрутный решатель OR-Tools)
  verify.py             независимая проверка точного решения: перебор и MILP в HiGHS
  routing.py            единый решатель маршрутов для всех методов
  methods/              пять методов распределения
  metrics.py            разрывы до эталона
  runner.py             расчёт одного примера: эталон, методы, показатели → JSON
  analysis.py           правило выбора метода и сравнение методов в ячейках
  stats.py, assign.py   статистические критерии; назначение на минимум суммы и максимума
experiments/            сценарии расчётов (запуск: python -m experiments.<имя>)
scripts/                то же для Windows двойным щелчком
data/                   итог расчётов, по которому построена статья
tests/                  проверки (322 теста, около 3 минут: python -m pytest -q)
```

## Методы

| Метод | Файл | Источник | Как реализован |
|---|---|---|---|
| Жадный алгоритм | `methods/greedy.py` | — | аппарат с наименьшим пройденным путём берёт ближайшую свободную задачу; вариант «по очереди» — как в [5] |
| CBBA | `methods/cbba.py` | Choi, Brunet, How [1] | синхронный, полносвязная сеть; ставка — прирост длины пути; варианты с ценностью задачи, убывающей со временем прибытия |
| k-means + венгерский метод | `methods/kmeans_hungarian.py` | Kuhn [2] | M кластеров k-means, кластеры назначаются аппаратам венгерским методом |
| Метод секторов | `methods/skfu.py` | Петренко и др. [4, 5] | по тексту статей; деление поля на секторы — по опубликованному коду авторов |
| GACA | `methods/gaca.py` | Rodriguez и др. [3] | по тексту препринта с уточнениями авторов; выбранные прочтения перечислены в начале файла |

После распределения маршрут каждого аппарата строится заново одним решателем (`routing.py`):
до 12 задач — точно, больше — детерминированный локальный поиск. Так сравнивается именно
распределение, а не способ построения маршрутов.

## Полный расчёт с нуля

Данные в `data/` получены так. Каждый шаг можно прервать и запустить снова той же командой:
посчитанные примеры пропускаются. Время указано для компьютера с 12 логическими ядрами.

| Шаг | Команда (Windows) | Что считает | Время |
|---|---|---|---|
| 1 | `scripts\run_exact_limits_windows.bat` | предел точного решения (раздел 2) | несколько часов |
| 2 | `scripts\run_experiment_windows.bat 50` | основной эксперимент, трёхмерная постановка, база в центре | около 2 суток |
| 3 | `scripts\run_checks_windows.bat 50` | проверки: число скоплений, один тип в скоплении | около 2 часов |
| 4 | `python -m experiments.gaca_check --reps 100 --ref-time 2 --side 25,55` | сверка с GACA (табл. 5) | около 1 часа |
| 5 | `python -m experiments.skfu_check --reps 100` | сверка с методом секторов (раздел 5.7) | несколько минут |
| 6 | `python -m experiments.build_data` | собрать `data/` из `results/` | около 5 минут |
| 7 | `python -m experiments.article` | таблицы, рисунки и сверка чисел | около 5 минут |

На Linux вместо `.bat` выполняются команды `python -m experiments.run_main …`, записанные внутри
этих файлов.

Тестовые примеры полностью воспроизводимы: сиды выводятся из параметров примера через SHA-256
и одинаковы на любом компьютере. Методы тоже детерминированы. Эталон при расчёте заново может
немного отличаться, потому что решатели работают с лимитом времени и в несколько потоков;
поэтому числа статьи воспроизводятся точно из `data/`, а повторный расчёт с нуля даёт близкие,
но не побитово совпадающие значения.

Отдельные сценарии:

- `python -m experiments.hypotheses` — проверка гипотез Г1–Г5;
- `python -m experiments.analyze_main` — правило выбора метода и сравнение методов в каждой ячейке;
- `python -m experiments.figures_main` — рисунки;
- `python -m experiments.mechanism` — целостность скоплений и неравномерность загрузки;
- `python -m experiments.calibrate_levels`, `python -m experiments.figures_fields` — калибровка уровней скученности;
- `python -m experiments.build_uniform_reference` — таблица нормировки показателя R (уже лежит в `allocsim/data/`).

## Пример использования

```python
from allocsim import InstanceSpec, make_instance, main_field
from allocsim.methods import cbba, kmeans_hungarian
from allocsim.exact import solve

spec = InstanceSpec(field=main_field(2, 1000.0, n_agents=4, tasks_per_agent=3, level="L3"),
                    n_agents=4, start_mode="base", heterogeneous=False)
inst = make_instance(spec, rep=0)          # 12 точек, сильные скопления, общая база
print(inst.R)                              # показатель скученности

for method in (cbba, kmeans_hungarian):
    r = method(inst)
    print(r.name, r.S, r.Lam, r.idle)      # суммарный путь, время выполнения, простаивающие

best = solve(inst, "Lambda", time_limit=30)   # точный минимум времени выполнения
print(best.Lam, best.optimal)
```

## Как ссылаться

Сведения для ссылки — в файле [`CITATION.cff`](CITATION.cff); после публикации статьи туда будут
добавлены её выходные данные.

## Лицензия

MIT, см. [`LICENSE`](LICENSE).

## Литература

1. Choi H.-L., Brunet L., How J.P. Consensus-based decentralized auctions for robust task
   allocation // IEEE Transactions on Robotics. 2009. Vol. 25, No. 4. P. 912–926.
   DOI: 10.1109/TRO.2009.2022423.
2. Kuhn H.W. The Hungarian method for the assignment problem // Naval Research Logistics
   Quarterly. 1955. Vol. 2, No. 1–2. P. 83–97. DOI: 10.1002/nav.3800020109.
3. Rodriguez J., Koenig S., Dong W., Lu Q. Grouping auction-consensus algorithm for decentralized
   task allocation in multi-robot systems. Препринт arXiv:2608.15884. 2026.
4. Петренко В.И., Тебуева Ф.Б., Антонов В.О., Сакольчик А.В. Распределение задач
   в кластеризированном поле целей для гомогенных и гетерогенных групп БПЛА // Робототехника
   и техническая кибернетика. 2023. Т. 11, № 2. С. 99–109. DOI: 10.31776/RTCJ.11203.
5. Petrenko V., Tebueva F., Antonov V., Ryabtsev S., Pavlov A., Sakolchik A. Method and algorithm
   for task allocation in a heterogeneous group of UAVs in a clustered field of targets // Journal
   of King Saud University – Computer and Information Sciences. 2023. Vol. 35, No. 6. Art. 101580.
   DOI: 10.1016/j.jksuci.2023.101580.
6. Clark P.J., Evans F.C. Distance to nearest neighbor as a measure of spatial relationships in
   populations // Ecology. 1954. Vol. 35, No. 4. P. 445–453. DOI: 10.2307/1931034.
7. Thomas M. A generalization of Poisson's binomial limit for use in ecology // Biometrika. 1949.
   Vol. 36, No. 1/2. P. 18–25. DOI: 10.2307/2332526.
8. Perron L., Didier F. CP-SAT. Google OR-Tools, версия 9.15. https://developers.google.com/optimization
9. Huangfu Q., Hall J.A.J. Parallelizing the dual revised simplex method // Mathematical
   Programming Computation. 2018. Vol. 10. P. 119–142. DOI: 10.1007/s12532-017-0130-5.

---

## English summary

Code and data for the paper "Choosing a task allocation method for a group of unmanned vehicles
depending on point clustering and the quality criterion" (in Russian; bibliographic details will
be added after publication).

Five allocation methods (greedy, CBBA, k-means + Hungarian assignment, field sectors, GACA) are
compared against the Pareto front of total path length and makespan: exact (CP-SAT) for up to
24 tasks, best known solution beyond that. Point clustering is the main experimental factor;
12 500 test instances in total.

**Reproduce the tables and figures** (Python 3.11–3.13, about 5 minutes):

```bash
pip install -r requirements.txt
python -m experiments.article
```

Tables 1–5, Figures 1–4 and `report.md`, which checks every number quoted in the paper against
its recomputed value, are written to `results/article/`. The input data are in `data/`
(see `data/README.md` for file and column descriptions). Instances are generated from SHA-256
seeds and are identical on any machine; a full recomputation from scratch (`experiments.build_data`
lists the steps) takes about two days on 12 logical cores and gives close but not bit-identical
references because the solvers run with time limits.

License: MIT.
