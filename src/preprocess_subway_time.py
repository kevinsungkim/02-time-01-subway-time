from pathlib import Path

import pandas as pd


# 파일 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "sample" / "raw_subway_time_sample.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "sample" / "processed_subway_time_sample.csv"

# 실제 Raw Data의 호선명 표기를 분석용 표기로 통일
LINE_NAME_MAP = {
    "경의선": "경의중앙선",
    "공항철도 1호선": "공항철도",
}


# 분석에 사용할 시간대 기준 정의
TIME_PERIODS = {
    "아침출근": {
        "hours": ["06시-07시", "07시-08시", "08시-09시"],
        "time_start": "06:00:00",
        "time_end": "09:00:00",
    },
    "낮": {
        "hours": [
            "09시-10시",
            "10시-11시",
            "11시-12시",
            "12시-13시",
            "13시-14시",
            "14시-15시",
            "15시-16시",
            "16시-17시",
            "17시-18시",
        ],
        "time_start": "09:00:00",
        "time_end": "18:00:00",
    },
    "저녁퇴근": {
        "hours": ["18시-19시", "19시-20시", "20시-21시"],
        "time_start": "18:00:00",
        "time_end": "21:00:00",
    },
    "밤": {
        "hours": ["21시-22시", "22시-23시", "23시-24시"],
        "time_start": "21:00:00",
        "time_end": "23:59:59",
    },
}

FINAL_COLUMNS = [
    "month_start_date",
    "month_end_date",
    "year",
    "month",
    "line_name",
    "station_name",
    "time_period",
    "time_start",
    "time_end",
    "in_passengers",
    "out_passengers",
    "total_passengers",
]


def main() -> None:
    # Raw Data 불러오기
    raw = pd.read_csv(RAW_FILE, encoding="cp949")

    # 기본 컬럼명 정리
    raw = raw.rename(
        columns={
            "사용월": "use_month",
            "호선명": "line_name",
            "지하철역": "station_name",
        }
    )

    required_base_columns = {"use_month", "line_name", "station_name"}
    missing_base_columns = required_base_columns - set(raw.columns)
    if missing_base_columns:
        raise ValueError(f"필수 기본 컬럼이 없습니다: {sorted(missing_base_columns)}")

    raw["station_name"] = (
        raw["station_name"]
        .astype("string")
        .str.strip()
        .apply(lambda name: name if name.endswith("역") else f"{name}역")
    )
    raw["line_name"] = raw["line_name"].replace(LINE_NAME_MAP)
    raw["_source_order"] = range(len(raw))

    # 월 기준 날짜 컬럼 생성
    use_month = pd.to_datetime(
        raw["use_month"].astype("string").str.strip(),
        format="%Y%m",
        errors="raise",
    )
    raw["month_start_date"] = use_month.dt.to_period("M").dt.start_time
    raw["month_end_date"] = use_month.dt.to_period("M").dt.end_time.dt.normalize()
    raw["year"] = use_month.dt.year
    raw["month"] = use_month.dt.month

    # 시간대별 승하차 인원 집계
    required_time_columns = [
        f"{hour} {direction}인원"
        for period in TIME_PERIODS.values()
        for hour in period["hours"]
        for direction in ("승차", "하차")
    ]
    missing_time_columns = set(required_time_columns) - set(raw.columns)
    if missing_time_columns:
        raise ValueError(f"필수 시간대 컬럼이 없습니다: {sorted(missing_time_columns)}")

    raw[required_time_columns] = raw[required_time_columns].apply(
        pd.to_numeric,
        errors="raise",
    )

    period_frames = []
    for time_period, period in TIME_PERIODS.items():
        period_frame = raw[
            [
                "month_start_date",
                "month_end_date",
                "year",
                "month",
                "line_name",
                "station_name",
                "_source_order",
            ]
        ].copy()
        period_frame["time_period"] = time_period
        period_frame["time_start"] = period["time_start"]
        period_frame["time_end"] = period["time_end"]
        period_frame["in_passengers"] = raw[
            [f"{hour} 승차인원" for hour in period["hours"]]
        ].sum(axis=1)
        period_frame["out_passengers"] = raw[
            [f"{hour} 하차인원" for hour in period["hours"]]
        ].sum(axis=1)
        period_frames.append(period_frame)

    # 분석용 subway_time 데이터셋 생성
    subway_time = pd.concat(period_frames, ignore_index=True)
    subway_time["total_passengers"] = (
        subway_time["in_passengers"] + subway_time["out_passengers"]
    )
    subway_time = subway_time.sort_values(
        ["month_start_date", "time_start", "_source_order"],
        ignore_index=True,
    )
    subway_time = subway_time[FINAL_COLUMNS]

    date_columns = ["month_start_date", "month_end_date"]
    subway_time[date_columns] = subway_time[date_columns].apply(
        lambda column: column.dt.strftime("%Y-%m-%d")
    )

    # 결과 파일 저장
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    subway_time.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # 실행 결과 검증
    expected_periods = set(TIME_PERIODS)
    assert list(subway_time.columns) == FINAL_COLUMNS
    assert len(subway_time) == len(raw) * len(TIME_PERIODS)
    assert set(subway_time["time_period"]) == expected_periods
    assert set(subway_time["time_start"]) == {
        period["time_start"] for period in TIME_PERIODS.values()
    }
    assert (
        subway_time.loc[subway_time["time_period"] == "밤", "time_end"]
        == "23:59:59"
    ).all()
    assert (
        subway_time["total_passengers"]
        == subway_time["in_passengers"] + subway_time["out_passengers"]
    ).all()

    print("최종 row 수:", len(subway_time))
    print("time_period 값:", sorted(subway_time["time_period"].unique()))
    print(
        "밤 시간대 종료 시각:",
        subway_time.loc[
            subway_time["time_period"] == "밤", "time_end"
        ].unique(),
    )


if __name__ == "__main__":
    main()
