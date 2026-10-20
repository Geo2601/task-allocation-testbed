"""Сверка чисел статьи: форматирование и сравнение (experiments/article.py)."""
from experiments.article import Checks, f, pct, rng


def test_number_format_like_article():
    assert f(0.775, 2) == "0,78"          # половина — вверх, как принято в тексте
    assert f(0.8248, 2) == "0,82"
    assert f(99.25, 1) == "99,3"
    assert f(-0.3, 1) == "−0,3"           # типографский минус
    assert f(-0.0001, 1) == "0,0"         # без «минус нуля»
    assert pct(0.471) == "47"
    assert rng(0.003, 0.041, (1, 0), 100) == "0,3–4"


def test_checks_exact_and_tolerance():
    c = Checks()
    c.add("1", "точное совпадение", "5,4", f(5.4, 1))
    c.add("1", "расхождение", "5,4", f(5.5, 1))
    c.add("1", "в пределах допуска", "— 0,71 0,88", "— 0,72 0,88", tol=0.01)
    c.add("1", "вне допуска", "0,71", "0,73", tol=0.01)
    assert c.frame().совпадает.tolist() == [True, False, True, False]
