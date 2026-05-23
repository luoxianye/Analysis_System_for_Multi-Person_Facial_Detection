from analyzer import judge_classroom_status


def test_no_people():
    status = judge_classroom_status(
        total=0,
        counts={},
        ratios={},
        main_expression="None",
    )
    assert status == "未检测到学生"


def test_good_status():
    ratios = {
        "Happy": 0.4,
        "Neutral": 0.4,
        "Sad": 0.1,
        "Angry": 0.1,
        "Surprise": 0.0,
    }
    status = judge_classroom_status(
        total=10,
        counts={},
        ratios=ratios,
        main_expression="Neutral",
        good_threshold=0.7,
        low_threshold=0.4,
        surprise_threshold=0.3,
        high_neutral_threshold=0.6,
    )
    assert status == "课堂状态良好"


def test_low_status():
    ratios = {
        "Happy": 0.1,
        "Neutral": 0.2,
        "Sad": 0.3,
        "Angry": 0.2,
        "Surprise": 0.2,
    }
    status = judge_classroom_status(
        total=10,
        counts={},
        ratios=ratios,
        main_expression="Sad",
        good_threshold=0.7,
        low_threshold=0.4,
        surprise_threshold=0.3,
        high_neutral_threshold=0.6,
    )
    assert status == "课堂状态较低落或需要关注"
