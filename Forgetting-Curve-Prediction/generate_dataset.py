
# ----------------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------------
import argparse                                  # reads options typed on the command line
import math                                      # math.log for the interval formula and lognormal medians
from bisect import bisect_left, bisect_right     # fast search in sorted lists of school days
from datetime import date, datetime, time, timedelta   # calendar arithmetic (plain Python is fastest here)

import numpy as np                               # fast numeric arrays and random numbers
import pandas as pd                              # tables (DataFrames) and CSV writing

# ----------------------------------------------------------------------------------
# GLOBAL SETTINGS (shared with the other two files)
# ----------------------------------------------------------------------------------
RETENTION_REF = 0.9                  # the 0.9-form curve: S = days until recall falls to 90%
CONCEPT_TYPES = ["vocabulary", "formula", "procedure", "conceptual"]   # distinct concept types

# Smoothing constants used by student_avg_accuracy (Beta prior: 8 "virtual" successes, 2 failures).
ACC_PRIOR_SUCCESS = 8.0
ACC_PRIOR_FAIL = 2.0

# Smoothing strength used by concept_complexity (acts like k "virtual" average reviews).
COMPLEXITY_SMOOTHING_K = 20.0

# Final 13 model inputs. The ORDER here is the order the model sees.
NUMERIC_FEATURES = [
    "time_delta", "log_time_delta",
    "history_correct", "history_wrong",
    "scaled_correct", "scaled_wrong",
    "success_streak", "last_rating",
    "response_duration", "concept_complexity",
    "student_avg_accuracy", "retrievability",
]
CATEGORICAL_FEATURES = ["concept_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES   # 12 numeric + 1 categorical = 13
TARGET_COLUMN = "recalled"                                  # 1 = remembered, 0 = forgot


# ----------------------------------------------------------------------------------
# SHARED FEATURE FORMULAS (vectorised: work on numbers, numpy arrays or pandas columns)
# ----------------------------------------------------------------------------------
def scaled_count(count):
    """scaled_correct / scaled_wrong = sqrt(1 + count)  (diminishing returns, Duolingo HLR style)."""
    return np.sqrt(1.0 + count)


def log_time(time_delta_days):
    """log_time_delta = ln(1 + time_delta)  (compresses the very skewed review gaps)."""
    return np.log1p(time_delta_days)


def estimate_stability(scaled_correct, scaled_wrong, concept_complexity, last_rating, student_avg_accuracy):
    """
    estimated_stability S (in days, 0.9-form) from PAST information only.
    HLR-style log-linear model:  S = exp(theta . x)
    The theta weights are fixed, hand-chosen starting values (a transparent baseline,
    NOT the hidden simulator rule).  The ML model can then improve on this baseline.
    """
    log_s = (1.2                                   # intercept: about 3.3 days with no history
             + 0.60 * scaled_correct               # more successes   -> stronger memory
             - 0.35 * scaled_wrong                 # more failures    -> weaker memory
             - 1.00 * concept_complexity           # harder concept   -> weaker memory
             + 0.15 * (last_rating - 3.0)          # last rating above "Good" -> stronger
             + 1.00 * (student_avg_accuracy - 0.8))  # better student than average -> stronger
    return np.clip(np.exp(log_s), 0.05, 365.0)      # keep S inside a sane range (days)


def retrievability(time_delta_days, stability_days):
    """R(t, S) = 0.9 ** (t / S)  : probability of recall after t days with stability S."""
    return np.power(RETENTION_REF, time_delta_days / stability_days)


def interval_for_target(stability_days, target_retention):
    """t* = S * ln(r_target) / ln(0.9) : days until recall falls to target_retention."""
    return stability_days * math.log(target_retention) / math.log(RETENTION_REF)


def add_engineered_features(df):
    """
    Adds the derived columns to a table that already has the raw columns:
    time_delta, history_correct, history_wrong, last_rating, concept_complexity,
    student_avg_accuracy.  Returns a NEW DataFrame (the input is not modified).
    """
    out = df.copy()                                                     # never edit the caller's table
    out["log_time_delta"] = log_time(out["time_delta"])                 # ln(1 + t)
    out["scaled_correct"] = scaled_count(out["history_correct"])        # sqrt(1 + correct)
    out["scaled_wrong"] = scaled_count(out["history_wrong"])            # sqrt(1 + wrong)
    out["estimated_stability"] = estimate_stability(                    # baseline memory strength
        out["scaled_correct"], out["scaled_wrong"], out["concept_complexity"],
        out["last_rating"], out["student_avg_accuracy"])
    out["retrievability"] = retrievability(out["time_delta"], out["estimated_stability"])  # 0.9^(t/S)
    return out




# ----------------------------------------------------------------------------------
# US SCHOOL / COLLEGE WORLD (assumptions, not measured statistics)
# ----------------------------------------------------------------------------------
BAND_WEIGHTS = {"middle_school": 0.25, "high_school": 0.40, "college": 0.35}   # share of students
BAND_CALENDAR = {"middle_school": "k12", "high_school": "k12", "college": "college"}

# Teaching windows (inclusive). Gaps between windows are breaks: Thanksgiving, winter break,
# spring break, summer. Approximate typical dates for the 2025-26 year plus fall 2026.
CALENDARS = {
    "k12": [
        (date(2025, 8, 18), date(2025, 11, 21)), (date(2025, 12, 1), date(2025, 12, 19)),   # fall 2025
        (date(2026, 1, 5), date(2026, 3, 13)), (date(2026, 3, 23), date(2026, 5, 29)),      # spring 2026
        (date(2026, 8, 17), date(2026, 11, 20)), (date(2026, 11, 30), date(2026, 12, 18)),  # fall 2026
    ],
    "college": [
        (date(2025, 8, 25), date(2025, 11, 25)), (date(2025, 12, 1), date(2025, 12, 12)),   # fall 2025
        (date(2026, 1, 12), date(2026, 3, 6)), (date(2026, 3, 16), date(2026, 5, 1)),       # spring 2026
        (date(2026, 8, 24), date(2026, 11, 24)), (date(2026, 11, 30), date(2026, 12, 11)),  # fall 2026
    ],
}
TERM_WINDOWS = {"fall2025": (0, 1), "spring2026": (2, 3), "fall2026": (4, 5)}   # window index ranges

# Typical subjects per band: subject -> (probability of each concept type, extra difficulty).
# Order of types: vocabulary, formula, procedure, conceptual.
SUBJECTS = {
    "middle_school": {
        "Pre-Algebra":     ((0.05, 0.40, 0.40, 0.15), 0.00),
        "Life Science":    ((0.45, 0.05, 0.10, 0.40), 0.00),
        "US History":      ((0.40, 0.00, 0.00, 0.60), -0.05),
        "ELA Vocabulary":  ((0.85, 0.00, 0.05, 0.10), -0.10),
        "Spanish":         ((0.75, 0.00, 0.25, 0.00), 0.00),
    },
    "high_school": {
        "Algebra/Geometry": ((0.05, 0.40, 0.40, 0.15), 0.00),
        "Biology":          ((0.45, 0.05, 0.10, 0.40), 0.00),
        "US History":       ((0.40, 0.00, 0.00, 0.60), -0.05),
        "Chemistry":        ((0.15, 0.40, 0.25, 0.20), 0.10),
        "Spanish":          ((0.75, 0.00, 0.25, 0.00), 0.00),
        "ELA Vocabulary":   ((0.85, 0.00, 0.05, 0.10), -0.10),
    },
    "college": {
        "Calculus":          ((0.05, 0.40, 0.40, 0.15), 0.10),
        "Organic Chemistry": ((0.20, 0.05, 0.45, 0.30), 0.20),
        "Psychology":        ((0.50, 0.00, 0.00, 0.50), -0.05),
        "Anatomy":           ((0.70, 0.00, 0.05, 0.25), 0.05),
        "Economics":         ((0.15, 0.30, 0.10, 0.45), 0.05),
        "Spanish":           ((0.75, 0.00, 0.25, 0.00), 0.00),
    },
}
SUBJECTS_PER_STUDENT = 5                                       # courses each student takes

# Study habits (assumptions): chance a student is a "crammer", and the chance that a study
# wish that lands on a weekend really happens on the weekend (otherwise it moves to a weekday).
CRAM_PROB = {"middle_school": 0.35, "high_school": 0.45, "college": 0.55}
WEEKEND_PROB = {"middle_school": 0.35, "high_school": 0.40, "college": 0.60}
BREAK_STUDY_PROB = 0.05                                        # chance of studying on a break day

# Hour-of-day mixtures: (probability, start_hour, end_hour). Hours above 24 mean after midnight.
HOUR_MIX = {
    "k12": [(0.25, 8, 15), (0.10, 6, 8), (0.55, 15, 21), (0.10, 21, 23.5)],
    "college": [(0.10, 7, 9), (0.20, 9, 16), (0.35, 16, 21), (0.35, 21, 26)],
}

# Memory behaviour per concept type in the HIDDEN simulator:
#   s0 = starting stability (days), growth = how strongly successful reviews strengthen memory,
#   lapse = share of stability kept after a failed recall.
TYPE_PARAMS = {
    "vocabulary": {"s0": 2.6, "growth": 1.0, "lapse": 0.40},
    "formula":    {"s0": 3.2, "growth": 0.8, "lapse": 0.35},
    "procedure":  {"s0": 5.0, "growth": 1.3, "lapse": 0.50},
    "conceptual": {"s0": 3.8, "growth": 1.1, "lapse": 0.45},
}


class Calendar:
    """Helper that knows which days are teaching days (weekdays inside teaching windows)."""

    def __init__(self, windows):
        self.windows = windows
        self.weekdays = []                                         # sorted list of teaching weekdays
        for start, end in windows:
            day = start
            while day <= end:
                if day.weekday() < 5:                              # Monday..Friday
                    self.weekdays.append(day)
                day += timedelta(days=1)
        self.term_days = {name: [d for d in self.weekdays if windows[a][0] <= d <= windows[b][1]]
                          for name, (a, b) in TERM_WINDOWS.items()}
        self.end = windows[-1][1]                                  # last simulated day

    def in_window(self, day):
        """True if the date is inside any teaching window (weekends included)."""
        return any(start <= day <= end for start, end in self.windows)

    def next_weekday(self, day, jitter=0):
        """First teaching weekday on/after `day`, plus `jitter` extra teaching days. None if past the end."""
        i = bisect_left(self.weekdays, day) + jitter
        return self.weekdays[i] if i < len(self.weekdays) else None

    def prev_weekday(self, day):
        """Last teaching weekday on/before `day`. None if before the start."""
        i = bisect_right(self.weekdays, day) - 1
        return self.weekdays[i] if i >= 0 else None


CAL_OBJECTS = {name: Calendar(windows) for name, windows in CALENDARS.items()}   # built once


def sample_hour(rng, band):
    """Hour of day (can exceed 24 = after midnight) from the band's mixture."""
    mix = HOUR_MIX[BAND_CALENDAR[band]]
    probs = np.array([m[0] for m in mix])
    start, end = mix[rng.choice(len(mix), p=probs / probs.sum())][1:]
    return float(rng.uniform(start, end))


def snap_to_study_time(rng, cal, band, target, backward=False):
    """
    Moves a wished-for datetime to a realistic study time:
      - break day  -> usually moves to the first teaching day after the break (5% study anyway)
      - weekend    -> usually moves to a weekday (weekend studying with band-specific probability)
      - hour       -> sampled from the band's hour-of-day pattern
    backward=True (exam prep) moves to the PREVIOUS weekday so prep stays before the exam.
    Returns None if the time falls outside the simulated calendar.
    """
    day = target.date()
    if not cal.in_window(day):                                     # break or summer
        if rng.random() >= BREAK_STUDY_PROB:
            day = cal.prev_weekday(day) if backward else cal.next_weekday(day, int(rng.integers(0, 3)))
    elif day.weekday() >= 5:                                       # weekend inside a term
        if rng.random() >= WEEKEND_PROB[band]:
            day = cal.prev_weekday(day) if backward else cal.next_weekday(day)
    if day is None or day > cal.end:
        return None
    return datetime.combine(day, time(0)) + timedelta(hours=sample_hour(rng, band))


def schedule_reviews(rng, band, term, crammer, max_reviews):
    """
    Builds the first-exposure time and the review times of ONE student-concept pair.
    Reviews come from three realistic sources:
      regular     - homework / follow-up reviews after first learning
      exam_prep   - reviews just before a midterm or final (crammers leave them very late)
      cumulative  - reviews in the NEXT term (cumulative exam, next course, back from summer)
    Returns (intro_datetime, [(review_datetime, label), ...]) or (None, []) if too few reviews.
    """
    cal = CAL_OBJECTS[BAND_CALENDAR[band]]
    key = "fall2025" if term == "fall" else "spring2026"
    next_key = "spring2026" if term == "fall" else "fall2026"
    days = cal.term_days[key]
    length = len(days)

    final_exam = rng.random() < 0.5                                # final exam or midterm?
    exam_idx = length - 1 - int(rng.integers(0, 3)) if final_exam else int(length * rng.uniform(0.45, 0.55))
    exam_day = days[exam_idx]
    exam_dt = datetime.combine(exam_day, time(8))                  # exams assumed in the morning
    intro_idx = int(rng.integers(0, max(1, exam_idx - 6)))         # concept taught at least ~a week before
    intro = snap_to_study_time(rng, cal, band, datetime.combine(days[intro_idx], time(0)))
    if intro is None:
        return None, []

    events = []                                                    # (wished datetime, label, backward?)
    # 1) regular reviews after teaching
    median_gap = 6.0 if crammer else 2.5
    t = intro
    for _ in range(int(rng.integers(1, 4))):
        t = t + timedelta(days=float(rng.lognormal(math.log(median_gap), 0.6)))
        if t.date() >= exam_day:
            break
        events.append((t, "regular", False))
    # 2) exam preparation reviews
    mean_before = 1.5 if crammer else 3.5                          # days before the exam
    n_prep = 1 + int(rng.random() < 0.6) + int(rng.random() < 0.3)
    for _ in range(n_prep):
        t = exam_dt - timedelta(days=0.3 + float(rng.exponential(mean_before)))
        if t > intro + timedelta(days=1):
            events.append((t, "exam_prep", True))
    # 3) cumulative reviews in the next term (after winter break or summer)
    if rng.random() < (0.55 if final_exam else 0.45):
        next_days = cal.term_days[next_key]
        idx = min(int(rng.beta(1.2, 3.0) * len(next_days)), len(next_days) - 1)
        events.append((datetime.combine(next_days[idx], time(0)), "cumulative", False))
        if rng.random() < 0.4 and idx + 7 < len(next_days):
            events.append((datetime.combine(next_days[idx + int(rng.integers(1, 8))], time(0)), "cumulative", False))
    # 4) make sure there are at least 3 reviews
    while len(events) < 3:
        span = (exam_dt - intro).total_seconds()
        events.append((intro + timedelta(seconds=float(rng.uniform(0.1, 0.9) * span)), "regular", False))

    # Snap every wished time to a realistic study time, then sort and keep valid ones.
    final = []
    for wished, label, backward in events:
        snapped = snap_to_study_time(rng, cal, band, wished, backward)
        if snapped is not None:
            final.append((snapped, label))
    final.sort(key=lambda e: e[0])
    cleaned, last = [], intro
    for when, label in final:                                      # at least 1 hour between events
        if when < last + timedelta(hours=1):
            when = last + timedelta(hours=float(rng.uniform(1, 4)))
        cleaned.append((when, label))
        last = when
    cleaned = cleaned[:max_reviews]
    return (intro, cleaned) if len(cleaned) >= 3 else (None, [])


# ----------------------------------------------------------------------------------
# THE HIDDEN MEMORY SIMULATOR (what "really" happens inside the synthetic students)
# ----------------------------------------------------------------------------------
def simulate_sequence(rng, ability, concept_type, true_complexity, intro, events):
    """
    Simulates the MEMORY of one student for one concept over the given review times.
    The review times come from the school calendar; they do NOT depend on memory strength.
    Returns one dict per review holding features known BEFORE the review, the outcome,
    and hidden ground truth (columns starting with gt_).
    """
    params = TYPE_PARAMS[concept_type]
    stability = params["s0"] * ability * (1.3 - true_complexity)   # starting stability (days)
    stability = float(np.clip(stability, 0.2, 30.0))

    history_correct, history_wrong, streak = 0, 0, 0
    last_rating = 3                                                # first teaching counts as "Good"
    prev_duration = float(np.clip(rng.lognormal(np.log(6.0 * (1 + true_complexity)), 0.25), 0.5, 60.0))
    prev_time = intro
    rows = []

    for k, (when, label) in enumerate(events, start=1):
        gap = max((when - prev_time).total_seconds() / 86400.0, 1.0 / 24.0)   # days since previous event
        p_true = float(0.9 ** (gap / stability))                   # hidden 0.9-form forgetting curve
        recalled = bool(rng.random() < p_true)

        rows.append({
            "review_index": k, "review_timestamp": when, "review_context": label,
            "time_delta": gap,
            "history_correct": history_correct, "history_wrong": history_wrong,
            "success_streak": streak, "last_rating": last_rating,
            "response_duration": prev_duration,                    # duration of the PREVIOUS review
            "recalled": int(recalled),
            "gt_true_stability": stability, "gt_true_recall_prob": p_true,
            "gt_true_complexity": true_complexity,
        })

        if not recalled:
            rating = 1                                             # Again
        else:
            difficulty = (1.0 - p_true) + rng.normal(0.0, 0.1)
            rating = 2 if difficulty > 0.30 else (3 if difficulty > 0.10 else 4)
        duration = 3.0 * (1.0 + 2.5 * (1.0 - p_true)) * (1.0 + 0.5 * true_complexity)
        duration *= 1.5 if not recalled else 1.0
        duration = float(np.clip(duration * float(rng.lognormal(0.0, 0.25)), 0.5, 60.0))

        if recalled:
            bonus = {2: 0.8, 3: 1.0, 4: 1.3}[rating]
            gain = 1.0 + 3.0 * params["growth"] * (1.2 - true_complexity) * (stability ** -0.2) * \
                (np.exp(1.0 - p_true) - 1.0 + 0.25) * bonus        # low recall before = bigger gain
            stability = float(min(stability * gain, 365.0))
            history_correct += 1
            streak += 1
        else:
            stability = float(max(stability * params["lapse"], 0.2))
            history_wrong += 1
            streak = 0

        last_rating, prev_duration, prev_time = rating, duration, when
    return rows


def build_concept_catalogue(rng, concepts_per_subject):
    """Creates concepts for every (grade band, subject): id, type and hidden complexity."""
    catalogue, next_id = [], 0
    for band, subjects in SUBJECTS.items():
        for subject, (type_probs, extra_difficulty) in subjects.items():
            for _ in range(concepts_per_subject):
                ctype = str(rng.choice(CONCEPT_TYPES, p=type_probs))
                base = rng.beta(2.0, 2.0) * 0.9 + 0.05
                catalogue.append({"concept_id": f"C{next_id:04d}", "band": band, "subject": subject,
                                  "concept_type": ctype,
                                  "true_complexity": float(np.clip(base + extra_difficulty, 0.05, 0.95))})
                next_id += 1
    return pd.DataFrame(catalogue)


def generate_dataset(n_students, concepts_per_subject, concepts_per_student, max_reviews, seed, target_retention):
    """Builds the complete synthetic DataFrame with all documented features and targets."""
    rng = np.random.default_rng(seed)
    catalogue = build_concept_catalogue(rng, concepts_per_subject)
    bands = list(BAND_WEIGHTS)
    band_p = np.array(list(BAND_WEIGHTS.values()))

    all_rows = []
    for sid in range(n_students):
        band = str(rng.choice(bands, p=band_p))                    # middle school, high school or college
        ability = float(rng.lognormal(0.0, 0.45))                  # memory ability multiplier
        crammer = bool(rng.random() < CRAM_PROB[band])             # leaves studying to the last days?
        subject_names = list(SUBJECTS[band])
        chosen_subjects = rng.choice(subject_names, size=min(SUBJECTS_PER_STUDENT, len(subject_names)), replace=False)
        pool = catalogue[(catalogue["band"] == band) & (catalogue["subject"].isin(chosen_subjects))]
        picked = pool.sample(n=min(concepts_per_student, len(pool)),
                             random_state=int(rng.integers(0, 2**31 - 1)))
        for concept in picked.itertuples(index=False):
            term = "fall" if rng.random() < 0.5 else "spring"
            intro, events = schedule_reviews(rng, band, term, crammer, max_reviews)
            if intro is None:
                continue
            for row in simulate_sequence(rng, ability, concept.concept_type, concept.true_complexity, intro, events):
                row.update({"student_id": f"S{sid:05d}", "concept_id": concept.concept_id,
                            "concept_type": concept.concept_type, "subject": concept.subject,
                            "grade_band": band, "term": term})
                all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # ---- student_avg_accuracy: expanding, smoothed recall rate using PAST reviews only
    df = df.sort_values(["student_id", "review_timestamp"], kind="stable").reset_index(drop=True)
    by_student = df.groupby("student_id")
    prior_success = by_student["recalled"].cumsum() - df["recalled"]
    prior_count = by_student.cumcount()
    df["student_avg_accuracy"] = (prior_success + ACC_PRIOR_SUCCESS) / (prior_count + ACC_PRIOR_SUCCESS + ACC_PRIOR_FAIL)

    # ---- Split by STUDENT (not by row): 70% train, 15% valid, 15% test
    shuffled = rng.permutation(df["student_id"].unique())
    n_train, n_valid = int(0.70 * len(shuffled)), int(0.15 * len(shuffled))
    split_of = {s: "train" for s in shuffled[:n_train]}
    split_of.update({s: "valid" for s in shuffled[n_train:n_train + n_valid]})
    split_of.update({s: "test" for s in shuffled[n_train + n_valid:]})
    df["split"] = df["student_id"].map(split_of)

    # ---- concept_complexity: smoothed ERROR RATE per concept, TRAIN rows only (no leakage)
    train = df[df["split"] == "train"]
    global_error = 1.0 - train["recalled"].mean()
    stats = train.groupby("concept_id")["recalled"].agg(["count", "sum"])
    wrong = stats["count"] - stats["sum"]
    smoothed = (wrong + COMPLEXITY_SMOOTHING_K * global_error) / (stats["count"] + COMPLEXITY_SMOOTHING_K)
    df["concept_complexity"] = df["concept_id"].map(smoothed).fillna(global_error)

    # ---- Derived features (scaled counts, log time, stability estimate, retrievability)
    df = add_engineered_features(df)

    # ---- Derived label: optimal review interval from the HIDDEN true stability
    df["optimal_review_interval"] = df["gt_true_stability"].apply(lambda s: interval_for_target(s, target_retention))

    ordered = (["student_id", "concept_id", "review_timestamp", "review_index", "split",
                "grade_band", "subject", "term", "review_context"]
               + FEATURE_COLUMNS + ["estimated_stability"]
               + [TARGET_COLUMN, "optimal_review_interval"]
               + ["gt_true_stability", "gt_true_recall_prob", "gt_true_complexity"])
    return df[ordered]


# ----------------------------------------------------------------------------------
# BUILT-IN AUDIT (runs automatically after the data is generated)
#   FLOW   checks must PASS (they prove the rows follow the forgetting-curve logic)
#   REAL   checks WARN if the data looks unusual versus the US-calendar assumptions above
#   SIGNAL checks WARN if the features carry too little information
# ----------------------------------------------------------------------------------
def run_audit(df):
    """Prints the audit table. Returns the number of hard FLOW failures."""
    from sklearn.metrics import roc_auc_score                      # imported here so the generator itself needs only numpy/pandas
    d = df.sort_values(["student_id", "concept_id", "review_timestamp"]).reset_index(drop=True)
    g = d.groupby(["student_id", "concept_id"])
    results = []

    def check(level, name, ok, detail=""):
        status = "PASS" if ok else ("FAIL" if level == "FLOW" else "WARN")
        results.append(status)
        print(f"  [{status}] {name} {detail}")

    print("\nAUDIT A) ALGORITHM FLOW")
    check("FLOW", "history_correct + history_wrong = review_index - 1",
          bool((d.history_correct + d.history_wrong == d.review_index - 1).all()))
    check("FLOW", "history_correct = running count of past successes",
          bool((d.history_correct == g.recalled.cumsum() - d.recalled).all()))
    check("FLOW", "success_streak <= history_correct", bool((d.success_streak <= d.history_correct).all()))
    prev = g.recalled.shift(1)
    check("FLOW", "last_rating = 1 exactly when previous review failed",
          bool(((d.last_rating == 1)[prev.notna()] == (prev[prev.notna()] == 0)).all()))
    later = d.review_index > 1
    gap_clock = g.review_timestamp.diff().dt.total_seconds() / 86400
    check("FLOW", "time_delta equals gap between consecutive timestamps",
          bool((gap_clock[later] - d.time_delta[later]).abs().max() < 1e-3))
    check("FLOW", "timestamps strictly increase inside each student-concept",
          bool((g.review_timestamp.diff().dropna() > pd.Timedelta(0)).all()))
    check("FLOW", "every student belongs to exactly one split", bool(d.groupby("student_id").split.nunique().max() == 1))
    bins = pd.cut(d.time_delta / d.gt_true_stability, [0, .5, 1, 2, 4, 1e9])
    gap = (d.groupby(bins, observed=True).recalled.mean() - d.groupby(bins, observed=True).gt_true_recall_prob.mean()).abs().max()
    check("FLOW", "empirical recall matches the 0.9^(t/S) curve", bool(gap < 0.03), f"(max gap {gap:.3f})")
    check("FLOW", "time_delta positive and below 365 days", bool((d.time_delta > 0).all() and (d.time_delta < 365).all()),
          f"(max {d.time_delta.max():.0f} d)")

    print("AUDIT B) US SCHOOL / COLLEGE REALISM (assumption-based)")
    recall = d.recalled.mean()
    check("REAL", "overall recall between 0.60 and 0.90", 0.60 <= recall <= 0.90, f"({recall:.3f})")
    night = float((d.review_timestamp.dt.hour < 5).mean())
    check("REAL", "few reviews between midnight and 5am (< 12%)", night < 0.12, f"({night:.3f})")
    weekend = float(d.review_timestamp.dt.dayofweek.isin([5, 6]).mean())
    check("REAL", "weekend share well below uniform 28.6% (8%-24%)", 0.08 <= weekend <= 0.24, f"({weekend:.3f})")
    on_break = float(np.mean([not CAL_OBJECTS[BAND_CALENDAR[b]].in_window(t.date())
                              for b, t in zip(d.grade_band, d.review_timestamp)]))
    check("REAL", "reviews on break days are rare (< 10%)", on_break < 0.10, f"({on_break:.3f})")
    span = g.review_timestamp.agg(lambda x: (x.max() - x.min()).days)
    check("REAL", "no concept reviewed over more than ~17 months", int(span.max()) <= 520, f"(max {int(span.max())} d)")
    share_cum = float((d.review_context == "cumulative").mean())
    check("REAL", "some reviews come after a break/next term (5%-40%)", 0.05 <= share_cum <= 0.40, f"({share_cum:.3f})")
    sr = d.groupby("student_id").recalled.mean()
    check("REAL", "students differ in recall rate (std >= 0.08)", float(sr.std()) >= 0.08, f"(std {sr.std():.3f})")
    tr = d.groupby("concept_type").recalled.mean()
    check("REAL", "concept types differ in recall (range >= 0.03)", float(tr.max() - tr.min()) >= 0.03,
          f"({tr.round(3).to_dict()})")
    print("  Recall by grade band:", d.groupby("grade_band").recalled.mean().round(3).to_dict())
    print("  Share by review_context:", d.review_context.value_counts(normalize=True).round(3).to_dict())

    print("AUDIT C) SIGNAL IN THE FEATURES")
    hc = d.groupby(d.history_correct.clip(upper=6)).recalled.mean()
    check("SIGNAL", "recall rises with successful history (6+ vs 0 by >= 0.05)", float(hc.iloc[-1] - hc.iloc[0]) >= 0.05,
          f"(0: {hc.iloc[0]:.3f}, 6+: {hc.iloc[-1]:.3f})")
    tb = d.groupby(pd.cut(d.time_delta, [0, 1, 3, 7, 30, 400]), observed=True).recalled.mean()
    check("SIGNAL", "recall falls as the gap grows (first bin - last bin >= 0.15)", float(tb.iloc[0] - tb.iloc[-1]) >= 0.15,
          f"({tb.round(3).tolist()})")
    corr = float(d.concept_complexity.corr(d.gt_true_complexity))
    check("SIGNAL", "engineered concept_complexity tracks hidden complexity (corr >= 0.4)", corr >= 0.4, f"(corr {corr:.2f})")
    print(f"  Best possible AUC: {roc_auc_score(d.recalled, d.gt_true_recall_prob):.3f} | "
          f"retrievability-only AUC: {roc_auc_score(d.recalled, d.retrievability):.3f}")
    fails, warns = results.count("FAIL"), results.count("WARN")
    print(f"AUDIT SUMMARY: {results.count('PASS')} PASS, {warns} WARN, {fails} FAIL")
    return fails


# ----------------------------------------------------------------------------------
# COMMAND LINE ENTRY POINT
# ----------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic US-school forgetting-curve dataset.")
    parser.add_argument("--students", type=int, default=1000, help="number of students")
    parser.add_argument("--concepts_per_subject", type=int, default=12, help="concepts in each subject")
    parser.add_argument("--concepts_per_student", type=int, default=20, help="concepts each student studies")
    parser.add_argument("--max_reviews", type=int, default=10, help="max reviews per student-concept")
    parser.add_argument("--target_retention", type=float, default=0.9, help="retention for the interval label")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--out", type=str, default="synthetic_forgetting_dataset.csv", help="output CSV path")
    parser.add_argument("--skip_audit", action="store_true", help="do not run the built-in dataset audit")
    args = parser.parse_args()

    df = generate_dataset(args.students, args.concepts_per_subject, args.concepts_per_student,
                          args.max_reviews, args.seed, args.target_retention)
    df.to_csv(args.out, index=False)

    print(f"Saved {len(df):,} review rows to {args.out}")
    print("Rows per split:", df["split"].value_counts().to_dict())
    print("Students per grade band:", df.groupby("grade_band")["student_id"].nunique().to_dict())
    print(f"Overall recall rate: {df['recalled'].mean():.3f}")
    print("Median time_delta (days):", round(float(df['time_delta'].median()), 2))
    print("NOTE: SIMULATED data with ASSUMED US-calendar patterns, not measured US student statistics.")

    if not args.skip_audit:                                        # built-in realism + flow audit
        if run_audit(df) > 0:                                      # a FLOW failure means a bug in the generator
            raise SystemExit("Audit found FLOW failures: the dataset is not consistent. Do not train on it.")


if __name__ == "__main__":
    main()
