"""Учебные функции по лекциям и семинарам 7–8.

Никаких подключений к базе и персональных параметров.
diff_ci использует нормальное приближение для больших независимых групп.
cuped принимает выровненные pandas.Series без пропусков.
"""
import math
from statistics import NormalDist


def _design(sigma, alpha, power, comparisons):
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma должно быть положительным конечным числом")
    if not 0 < alpha < 1 or not 0.5 < power < 1:
        raise ValueError("Нужны 0 < alpha < 1 и 0.5 < power < 1")
    if not isinstance(comparisons, int) or isinstance(comparisons, bool) or comparisons < 1:
        raise ValueError("comparisons должно быть положительным целым")
    z = NormalDist().inv_cdf
    return z(1 - alpha / (2 * comparisons)) + z(power)


def n_per_group(sigma, mde, alpha=0.05, power=0.8, comparisons=1):
    """Размер КАЖДОЙ равной группы; абсолютный двусторонний MDE."""
    if not math.isfinite(mde) or mde <= 0:
        raise ValueError("mde должно быть положительным конечным числом")
    zsum = _design(sigma, alpha, power, comparisons)
    return math.ceil(2 * zsum ** 2 * sigma ** 2 / mde ** 2)


def mde_for_n(sigma, n, alpha=0.05, power=0.8, comparisons=1):
    """Абсолютный MDE при n независимых наблюдениях в КАЖДОЙ группе."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("n должно быть целым не меньше 2")
    zsum = _design(sigma, alpha, power, comparisons)
    return zsum * sigma * math.sqrt(2 / n)


def diff_ci(t, c, alpha=0.05):
    """test минус control: оценка, нормальный CI и нормальное двустороннее p.

    Средние, дисперсии и размеры считаются по одним и тем же наблюдениям:
    NA не удаляются молча. Нулевые ЗНАЧЕНИЯ, напротив, сохраняются.
    Для точного критерия Уэлча используйте scipy.stats.ttest_ind.
    """
    if not 0 < alpha < 1 or min(len(t), len(c)) < 2:
        raise ValueError("Проверьте alpha и размеры групп")
    if t.isna().any() or c.isna().any():
        raise ValueError("Сначала задайте явную политику обработки пропусков")
    if not all(math.isfinite(float(v)) for series in (t, c) for v in series):
        raise ValueError("Метрика должна содержать конечные числа")
    d = float(t.mean() - c.mean())
    se = math.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    if se == 0:
        raise ValueError("Нулевая стандартная ошибка: обычное приближение неприменимо")
    q = NormalDist().inv_cdf(1 - alpha / 2)
    p = math.erfc(abs(d / se) / math.sqrt(2))
    return {"diff": d, "se": se, "lo": d - q * se, "hi": d + q * se, "p": p}


def cuped(y, x):
    """Общая линейная поправка и theta; x измерен ДО назначения."""
    if len(y) < 2 or not y.index.equals(x.index):
        raise ValueError("Нужны одинаковые индексы и не меньше двух наблюдений")
    if y.isna().any() or x.isna().any():
        raise ValueError("Сначала обработайте пропуски по единому правилу")
    if not all(math.isfinite(float(v)) for series in (y, x) for v in series):
        raise ValueError("Ковариата и метрика должны быть конечными")
    vx = x.var(ddof=1)
    if vx <= 0:
        raise ValueError("Постоянная ковариата не подходит для CUPED")
    theta = float(y.cov(x) / vx)
    return y - theta * (x - x.mean()), theta


def srm_pvalue(sizes):
    """Chi-square SRM для РАВНЫХ долей двух или трёх групп, без scipy."""
    sizes = list(sizes)
    if len(sizes) not in (2, 3):
        raise ValueError("Эта формула реализована только для 2 или 3 групп")
    if any(not math.isfinite(float(s)) or s < 0 or int(s) != s for s in sizes):
        raise ValueError("sizes должны быть неотрицательными целыми частотами")
    n = sum(sizes)
    if n <= 0:
        raise ValueError("Пустая выборка")
    expected = n / len(sizes)
    chi2 = sum((s - expected) ** 2 / expected for s in sizes)
    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))
    return math.exp(-chi2 / 2)
