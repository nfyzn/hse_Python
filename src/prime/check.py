import pandas as pd

def find_missing_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Клиенты без канала привлечения."""
    return df[df["acquisition_channel"].isna()]


def find_reversed_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Сессии, которые кончились раньше, чем начались."""
    return df[df["ended_at"] < df["started_at"]]


def find_duplicate_payments(df: pd.DataFrame, gap_minutes: int = 20) -> pd.DataFrame:
    """Повторные списания по одной подписке на одну сумму в пределах gap_minutes."""
    s = df.sort_values(["subscription_id", "amount", "paid_at"])
    gap = s.groupby(["subscription_id", "amount"])["paid_at"].diff()
    return s[gap < pd.Timedelta(minutes=gap_minutes)]


print("три есть, четвёртую пишете вы")
