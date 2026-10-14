"""
synthetic_dataset_generation.py
================================
Generates the FIVE source datasets for the Spaced Repetition Optimization project
(US high schools, grades 9-12, and undergraduate colleges) and saves them in ./data/

    data/student_retention.csv   review-event log (one row per review attempt)
    data/course.csv              course offerings (course_id = course + term + section)
    data/topic.csv               topics / standards taught inside each course
    data/institution.csv         schools and colleges (fictional names)
    data/term.csv                academic calendars (finals, breaks)

Merging and preprocessing happen in dataset_processing.py (creates data/training.csv).

ALL DATA IS SYNTHETIC: names, IDs, scores and recall outcomes are simulated and must
not be presented as observed US student records.

Usage:
    pip install pandas numpy
    python synthetic_dataset_generation.py
    python synthetic_dataset_generation.py --seed 7 --students-per-institution 200
"""
import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class Config:
    seed: int = 42
    academic_year_start: int = 2025          # Fall 2025 -> Spring 2026
    n_high_schools: int = 4
    n_colleges: int = 4
    students_per_institution: int = 120
    sections_per_course: int = 2
    recall_threshold: float = 0.80           # items_correct / items_attempted
    max_hints_for_success: int = 1           # recall_success rule: hints_used <= this
    abandoned_session_rate: float = 0.005    # items_attempted = 0 -> recall_success null
    out_dir: str = str(Path(__file__).resolve().parent / "data")


# Memory-model constants (forgetting curve R = exp(-t / S), S = stability in days)
GUESS_FLOOR = 0.15
BASE_STABILITY_DAYS = 9.0
ACTIVITIES = ["Flashcard", "Practice Question", "Retrieval Quiz"]
ACTIVITY_P = [0.45, 0.30, 0.25]
ACTIVITY_ITEMS = {"Flashcard": (6, 17), "Practice Question": (4, 11), "Retrieval Quiz": (5, 13)}
ACTIVITY_GAIN = {"Flashcard": 1.00, "Practice Question": 1.05, "Retrieval Quiz": 1.15}
SPACING_BASE_DAYS = [1, 3, 7, 14, 21, 28, 35, 45]   # typical expanding schedule

# =============================================================================
# 2. REFERENCE DATA
# =============================================================================
STATE_TZ = {
    "TX": "America/Chicago", "CA": "America/Los_Angeles", "NY": "America/New_York",
    "FL": "America/New_York", "IL": "America/Chicago", "OH": "America/New_York",
    "WA": "America/Los_Angeles", "GA": "America/New_York", "AZ": "America/Phoenix",
    "CO": "America/Denver", "NC": "America/New_York", "PA": "America/New_York",
}
PLACE_NAMES = ["Riverbend", "Lakeview", "Oakridge", "Summit", "Cedar Valley", "Harborview",
               "Prairie Hills", "Mesa Verde", "Pinecrest", "Westfield", "Northgate", "Brookside",
               "Stonebridge", "Maple Grove", "Eastport", "Redwood"]
LOCALES = ["City", "Suburb", "Town", "Rural"]
LOCALE_P = [0.35, 0.35, 0.15, 0.15]

# HS topic tuple: (topic_name, standard_code, cognitive_level, unit_number, base_difficulty 0-1)
HS_TEMPLATES = [
    dict(code="ALG1", title="Algebra I", subject="Mathematics", levels=["Regular", "Honors"], ctype="Core", credit=0.5, topics=[
        ("Solving Linear Equations and Inequalities", "CCSS.MATH.CONTENT.HSA-REI.B.3", "Apply", 1, 0.30),
        ("Interpreting Linear Functions", "CCSS.MATH.CONTENT.HSF-IF.A.1", "Understand", 2, 0.40),
        ("Systems of Linear Equations", "CCSS.MATH.CONTENT.HSA-REI.C.6", "Apply", 3, 0.50),
        ("Linear vs. Exponential Models", "CCSS.MATH.CONTENT.HSF-LE.A.1", "Analyze", 4, 0.60),
        ("Polynomial Operations", "CCSS.MATH.CONTENT.HSA-APR.A.1", "Apply", 5, 0.55),
        ("Solving Quadratic Equations", "CCSS.MATH.CONTENT.HSA-REI.B.4", "Apply", 6, 0.70)]),
    dict(code="GEOM", title="Geometry", subject="Mathematics", levels=["Regular", "Honors"], ctype="Core", credit=0.5, topics=[
        ("Geometric Definitions and Constructions", "CCSS.MATH.CONTENT.HSG-CO.A.1", "Remember", 1, 0.30),
        ("Triangle Congruence Criteria", "CCSS.MATH.CONTENT.HSG-CO.B.8", "Understand", 2, 0.50),
        ("Similarity and Proportional Reasoning", "CCSS.MATH.CONTENT.HSG-SRT.B.5", "Apply", 3, 0.55),
        ("Right-Triangle Trigonometry", "CCSS.MATH.CONTENT.HSG-SRT.C.6", "Apply", 4, 0.65),
        ("Circle Theorems", "CCSS.MATH.CONTENT.HSG-C.A.2", "Analyze", 5, 0.65),
        ("Volume Formulas", "CCSS.MATH.CONTENT.HSG-GMD.A.3", "Apply", 6, 0.45)]),
    dict(code="BIO", title="Biology", subject="Science", levels=["Regular", "Honors"], ctype="Core", credit=0.5, topics=[
        ("DNA Structure and Protein Synthesis", "NGSS.HS-LS1-1", "Understand", 1, 0.60),
        ("Photosynthesis", "NGSS.HS-LS1-5", "Understand", 2, 0.50),
        ("Cellular Respiration", "NGSS.HS-LS1-7", "Understand", 3, 0.60),
        ("Inheritance of Traits", "NGSS.HS-LS3-1", "Apply", 4, 0.55),
        ("Natural Selection and Evolution", "NGSS.HS-LS4-2", "Analyze", 5, 0.45),
        ("Ecosystem Carrying Capacity", "NGSS.HS-LS2-1", "Analyze", 6, 0.40)]),
    dict(code="CHEM", title="Chemistry", subject="Science", levels=["Regular", "Honors"], ctype="Core", credit=0.5, topics=[
        ("Periodic Table Trends", "NGSS.HS-PS1-1", "Understand", 1, 0.40),
        ("Types of Chemical Reactions", "NGSS.HS-PS1-2", "Apply", 2, 0.50),
        ("Conservation of Mass and Stoichiometry", "NGSS.HS-PS1-7", "Apply", 3, 0.75),
        ("Reaction Rates", "NGSS.HS-PS1-5", "Analyze", 4, 0.60),
        ("Energy in Chemical Reactions", "NGSS.HS-PS1-4", "Analyze", 5, 0.65),
        ("Chemical Equilibrium", "NGSS.HS-PS1-6", "Analyze", 6, 0.80)]),
    dict(code="ENG10", title="English 10", subject="English Language Arts", levels=["Regular", "Honors"], ctype="Core", credit=0.5, topics=[
        ("Citing Textual Evidence", "CCSS.ELA-LITERACY.RL.9-10.1", "Apply", 1, 0.30),
        ("Determining Central Ideas", "CCSS.ELA-LITERACY.RI.9-10.2", "Understand", 2, 0.40),
        ("Analyzing Character Development", "CCSS.ELA-LITERACY.RL.9-10.3", "Analyze", 3, 0.50),
        ("Vocabulary in Context", "CCSS.ELA-LITERACY.L.9-10.4", "Understand", 4, 0.35),
        ("Writing Arguments", "CCSS.ELA-LITERACY.W.9-10.1", "Apply", 5, 0.60),
        ("Revising and Editing", "CCSS.ELA-LITERACY.W.9-10.5", "Apply", 6, 0.45)]),
    dict(code="USHIST", title="US History", subject="Social Studies", levels=["Regular", "Honors"], ctype="Core", credit=0.5, topics=[
        ("Colonial Origins and the American Revolution", "C3.D2.His.1.9-12", "Understand", 1, 0.35),
        ("The Constitution and Federalism", "C3.D2.Civ.1.9-12", "Understand", 2, 0.50),
        ("Westward Expansion", "C3.D2.His.2.9-12", "Analyze", 3, 0.40),
        ("Causes of the Civil War", "C3.D2.His.14.9-12", "Analyze", 4, 0.55),
        ("Reconstruction", "C3.D2.His.3.9-12", "Understand", 5, 0.50),
        ("Industrialization and Immigration", "C3.D2.His.16.9-12", "Analyze", 6, 0.45)]),
    dict(code="APCALCAB", title="AP Calculus AB", subject="Mathematics", levels=["AP"], ctype="Elective", credit=0.5, topics=[
        ("Limits and Continuity", "AP.CALCAB.U1", "Understand", 1, 0.55),
        ("Differentiation: Definition and Fundamental Properties", "AP.CALCAB.U2", "Apply", 2, 0.60),
        ("Differentiation: Composite, Implicit, and Inverse Functions", "AP.CALCAB.U3", "Apply", 3, 0.75),
        ("Contextual Applications of Differentiation", "AP.CALCAB.U4", "Analyze", 4, 0.70),
        ("Analytical Applications of Differentiation", "AP.CALCAB.U5", "Analyze", 5, 0.75),
        ("Integration and Accumulation of Change", "AP.CALCAB.U6", "Apply", 6, 0.70)]),
    dict(code="APSTAT", title="AP Statistics", subject="Mathematics", levels=["AP"], ctype="Elective", credit=0.5, topics=[
        ("Exploring One-Variable Data", "AP.STATS.U1", "Understand", 1, 0.35),
        ("Exploring Two-Variable Data", "AP.STATS.U2", "Analyze", 2, 0.45),
        ("Collecting Data", "AP.STATS.U3", "Understand", 3, 0.50),
        ("Probability and Random Variables", "AP.STATS.U4", "Apply", 4, 0.65),
        ("Sampling Distributions", "AP.STATS.U5", "Analyze", 5, 0.70),
        ("Inference for Proportions", "AP.STATS.U6", "Apply", 6, 0.75)]),
]

# College topic tuple: (topic_name, cognitive_level, base_difficulty); standards are course learning outcomes
_COLLEGE_RAW = [
    ("MATH101", "College Algebra", "Mathematics", "100-level", "Core", 3, [
        ("Functions and Graphs", "Understand", 0.35), ("Polynomial and Rational Functions", "Apply", 0.50),
        ("Exponential and Logarithmic Functions", "Apply", 0.60), ("Systems of Equations", "Apply", 0.45),
        ("Sequences and Series", "Analyze", 0.65), ("Conic Sections", "Analyze", 0.60)]),
    ("MATH151", "Calculus I", "Mathematics", "100-level", "Core", 4, [
        ("Limits", "Understand", 0.55), ("Derivative Rules", "Apply", 0.55), ("Applications of Derivatives", "Apply", 0.65),
        ("Optimization", "Analyze", 0.75), ("Definite Integrals", "Apply", 0.65), ("Fundamental Theorem of Calculus", "Analyze", 0.75)]),
    ("STAT201", "Introduction to Statistics", "Mathematics", "200-level", "Core", 3, [
        ("Descriptive Statistics", "Understand", 0.30), ("Probability Rules", "Apply", 0.55), ("Random Variables", "Apply", 0.60),
        ("Confidence Intervals", "Apply", 0.65), ("Hypothesis Testing", "Analyze", 0.75), ("Linear Regression", "Analyze", 0.70)]),
    ("CHEM111", "General Chemistry I", "Science", "100-level", "Core", 4, [
        ("Atomic Structure", "Understand", 0.45), ("Periodic Trends", "Understand", 0.40), ("Chemical Bonding", "Apply", 0.60),
        ("Stoichiometry", "Apply", 0.75), ("Gas Laws", "Apply", 0.60), ("Thermochemistry", "Analyze", 0.75)]),
    ("PSYC101", "Introduction to Psychology", "Social Sciences", "100-level", "Elective", 3, [
        ("Research Methods", "Understand", 0.40), ("Biological Bases of Behavior", "Remember", 0.50), ("Learning and Conditioning", "Understand", 0.45),
        ("Memory", "Understand", 0.45), ("Human Development", "Remember", 0.35), ("Social Psychology", "Understand", 0.35)]),
    ("ECON201", "Principles of Microeconomics", "Social Sciences", "200-level", "Core", 3, [
        ("Supply and Demand", "Understand", 0.40), ("Elasticity", "Apply", 0.60), ("Consumer Choice", "Apply", 0.65),
        ("Production and Costs", "Apply", 0.65), ("Perfect Competition", "Analyze", 0.60), ("Market Failures", "Analyze", 0.55)]),
    ("CSCI101", "Introduction to Programming", "Computer Science", "100-level", "Core", 3, [
        ("Variables and Data Types", "Remember", 0.25), ("Control Flow", "Apply", 0.40), ("Functions", "Apply", 0.55),
        ("Lists and Dictionaries", "Apply", 0.55), ("File Handling", "Apply", 0.50), ("Object-Oriented Basics", "Analyze", 0.70)]),
    ("BIOL220", "Anatomy and Physiology I", "Science", "200-level", "Core", 4, [
        ("Anatomical Terminology", "Remember", 0.40), ("Cell Structure", "Understand", 0.45), ("Tissues", "Remember", 0.55),
        ("Skeletal System", "Remember", 0.70), ("Muscular System", "Remember", 0.75), ("Nervous System", "Understand", 0.80)]),
]
COLLEGE_TEMPLATES = [
    dict(code=c, title=t, subject=s, levels=[lvl], ctype=ct, credit=cr,
         topics=[(n, f"LO-{c}-{i + 1:02d}", cog, i + 1, d) for i, (n, cog, d) in enumerate(tp)])
    for c, t, s, lvl, ct, cr, tp in _COLLEGE_RAW
]


# =============================================================================
# 3. CALENDAR HELPERS
# =============================================================================
def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th weekday (Mon=0) of a month; n=-1 means the last one."""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    first_next = date(year + (month == 12), month % 12 + 1, 1)
    d = first_next - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def holiday_list(year, college, spring_break_nth, fall_break, quarter):
    """(name, start, end) for the academic year that starts in `year`."""
    labor = nth_weekday(year, 9, 0, 1)
    thanks = nth_weekday(year, 11, 3, 4)
    mlk = nth_weekday(year + 1, 1, 0, 3)
    pres = nth_weekday(year + 1, 2, 0, 3)
    memorial = nth_weekday(year + 1, 5, 0, -1)
    hol = [("Labor Day", labor, labor),
           ("Thanksgiving Break", thanks - timedelta(days=1), thanks + timedelta(days=3 if college else 1)),
           ("MLK Day", mlk, mlk), ("Presidents Day", pres, pres), ("Memorial Day", memorial, memorial)]
    if not quarter:
        sb = nth_weekday(year + 1, 3, 0, spring_break_nth)
        hol.append(("Spring Break", sb, sb + timedelta(days=4)))
    if fall_break:
        fb = nth_weekday(year, 10, 0, 2)
        hol.append(("Fall Break", fb, fb + timedelta(days=1)))
    return hol


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def next_school_day(d: date, break_dates: set) -> date:
    while d.weekday() >= 5 or d in break_dates:
        d += timedelta(days=1)
    return d


# =============================================================================
# 4. INSTITUTIONS AND TERMS
# =============================================================================
def build_institutions(rng, cfg):
    rows = []
    n_total = cfg.n_high_schools + cfg.n_colleges
    names = list(rng.permutation(PLACE_NAMES))
    states = list(STATE_TZ)
    college_types = ["Community College", "Public 4-Year College", "Private 4-Year College"]
    for i in range(n_total):
        is_college = i >= cfg.n_high_schools
        state = states[int(rng.integers(len(states)))]
        place = names[i % len(names)]
        if is_college:
            itype = college_types[(i - cfg.n_high_schools) % 3]
            suffix = {"Community College": "Community College", "Public 4-Year College": "State University",
                      "Private 4-Year College": "College"}[itype]
            term_system = "Quarter" if rng.random() < 0.25 else "Semester"
            grade_range = "Undergraduate"
        else:
            itype = "Public High School" if rng.random() < 0.75 else "Private High School"
            suffix = "High School"
            term_system = "Semester"
            grade_range = "09-12"
        rows.append(dict(
            institution_id=f"INST-{i + 1:02d}", institution_name=f"{place} {suffix}",
            institution_type=itype, state_code=state,
            locale=str(rng.choice(LOCALES, p=LOCALE_P)),
            term_system=term_system, grade_range=grade_range, timezone=STATE_TZ[state],
            # internal helpers (not exported)
            _num=i + 1, _level="College" if is_college else "HS",
            _spring_break_nth=int(rng.integers(2, 4)), _fall_break=bool(rng.random() < 0.5)))
    return rows


def build_terms(rng, inst, year):
    college = inst["_level"] == "College"
    quarter = inst["term_system"] == "Quarter"
    hol = holiday_list(year, college, inst["_spring_break_nth"], inst["_fall_break"], quarter)

    if not college:   # high school semesters, ~17-18 instructional weeks
        fall_fin = nth_weekday(year, 12, 0, 2) + timedelta(weeks=int(rng.integers(0, 2)))
        spr_fin = nth_weekday(year + 1, 5, 0, int(rng.integers(2, 4)))
        specs = [("Fall", fall_fin, (18 if fall_fin.day >= 15 else 17) - int(rng.integers(0, 2))),
                 ("Spring", spr_fin, 17 + int(rng.integers(0, 2)))]
    elif not quarter:  # college semesters, 15 weeks
        specs = [("Fall", nth_weekday(year, 12, 0, 2), 15),
                 ("Spring", nth_weekday(year + 1, 5, 0, int(rng.integers(1, 3))), 15)]
    else:              # college quarters, 10 weeks
        specs = [("Fall", nth_weekday(year, 9, 0, 4) + timedelta(weeks=10), 10),
                 ("Winter", nth_weekday(year + 1, 1, 0, 1) + timedelta(weeks=10), 10),
                 ("Spring", nth_weekday(year + 1, 3, 0, 4) + timedelta(weeks=10), 10)]

    terms = []
    for name, fin_start, weeks in specs:
        instr_start = fin_start - timedelta(weeks=weeks)
        instr_end = fin_start - timedelta(days=3)           # Friday before finals
        fin_end = fin_start + timedelta(days=4)
        breaks = [(n, s, e) for n, s, e in hol if e >= instr_start and s <= fin_end]
        break_dates = {d for _, s, e in breaks for d in date_range(s, e)}
        instr_start = next_school_day(instr_start, break_dates)
        abbr = {"Fall": "FA", "Winter": "WI", "Spring": "SP"}[name]
        terms.append(dict(
            term_id=f"{inst['institution_id']}-{abbr}{str(fin_start.year)[2:]}",
            institution_id=inst["institution_id"], term_name=f"{name} {fin_start.year}",
            academic_year=f"{year}-{year + 1}", term_type="Quarter" if quarter else "Semester",
            term_start_date=instr_start - timedelta(days=7), term_end_date=fin_end + timedelta(days=2),
            instruction_start_date=instr_start, instruction_end_date=instr_end,
            finals_start_date=fin_start, finals_end_date=fin_end,
            break_periods=";".join(f"{s.isoformat()}/{e.isoformat()}" for _, s, e in breaks),
            _break_dates=break_dates, _quarter=quarter))
    return terms


# =============================================================================
# 5. COURSES AND TOPICS (exam dates are stored in course.exam_dates)
# =============================================================================
def build_catalog(rng, cfg, institutions, terms_by_inst):
    courses, topics = [], []
    offerings = {}       # (inst_id, term_id) -> {template_code: [course_id, ...]}
    course_topics = {}   # course_id -> [internal topic dicts]
    course_exams = {}    # course_id -> sorted [date]

    for inst in institutions:
        college = inst["_level"] == "College"
        templates = COLLEGE_TEMPLATES if college else HS_TEMPLATES
        instructors = [f"INS-{inst['_num']:02d}-{k:03d}" for k in range(1, 16)]
        level_of = {t["code"]: (str(rng.choice(t["levels"], p=[0.6, 0.4])) if len(t["levels"]) == 2 else t["levels"][0])
                    for t in templates}

        for term in terms_by_inst[inst["institution_id"]]:
            offerings[(inst["institution_id"], term["term_id"])] = {}
            for t in templates:
                level = level_of[t["code"]]
                title = f"Honors {t['title']}" if level == "Honors" else t["title"]
                for sec in range(1, cfg.sections_per_course + 1):
                    course_id = f"{term['term_id']}-{t['code']}-S{sec}"
                    offerings[(inst["institution_id"], term["term_id"])].setdefault(t["code"], []).append(course_id)
                    modality = str(rng.choice(["In-person", "Online", "Hybrid"],
                                              p=[0.65, 0.20, 0.15] if college else [0.95, 0.03, 0.02]))
                    courses.append(dict(
                        course_id=course_id, institution_id=inst["institution_id"], course_title=title,
                        subject_area=t["subject"], course_level=level, course_type=t["ctype"],
                        credit_value=t["credit"], modality=modality,
                        instructor_id=instructors[int(rng.integers(len(instructors)))]))

                    # ---- exams (supporting table) ----
                    start, end = term["instruction_start_date"], term["instruction_end_date"]
                    span = (end - start).days
                    fracs = [0.5] if term["_quarter"] else [0.33, 0.66]
                    ex_dates = []
                    for k, f in enumerate(fracs):
                        d = next_school_day(start + timedelta(days=int(span * f + rng.integers(-3, 4))), term["_break_dates"])
                        label = f"Unit Exam {k + 1}" if not college else (f"Midterm Exam {k + 1}" if len(fracs) > 1 else "Midterm Exam")
                        ex_dates.append((label, d))
                    ex_dates.append(("Final Exam", term["finals_start_date"] + timedelta(days=int(rng.integers(0, 5)))))
                    courses[-1]["exam_dates"] = ";".join(f"{label}:{d.isoformat()}" for label, d in ex_dates)
                    course_exams[course_id] = sorted(d for _, d in ex_dates)

                    # ---- topics ----
                    n_t = len(t["topics"])
                    internal = []
                    for i, (tname, std, cog, unit, base_d) in enumerate(t["topics"]):
                        topic_id = f"{course_id}-T{i + 1:02d}"
                        true_d = float(np.clip(base_d + rng.normal(0, 0.06) + (0.05 if level == "Honors" else 0), 0.05, 0.95))
                        off = int(round(i / n_t * 0.85 * span)) + int(rng.integers(0, 3))
                        idate = next_school_day(start + timedelta(days=off), term["_break_dates"])
                        mins = int(rng.integers(8, 25) * 5) if not college else int(rng.integers(10, 40) * 5)
                        est = int(round(mins * rng.uniform(0.9, 1.9) / 5) * 5)
                        # empirical difficulty = 1 - correct/answered, computed from HISTORICAL (prior-cohort) data only
                        n_hist = int(rng.integers(200, 1500))
                        p_ok = float(np.clip(0.90 - 0.60 * true_d + rng.normal(0, 0.03), 0.2, 0.97))
                        emp = round(1 - rng.binomial(n_hist, p_ok) / n_hist, 3)
                        topics.append(dict(
                            topic_id=topic_id, course_id=course_id, topic_name=tname, unit_number=unit,
                            standard_code=std, cognitive_level=cog, instruction_date=idate,
                            instruction_minutes=mins, empirical_difficulty=emp, estimated_learning_minutes=est))
                        internal.append(dict(topic_id=topic_id, course_id=course_id, term_id=term["term_id"],
                                             institution_id=inst["institution_id"], instruction_date=idate,
                                             instruction_minutes=mins, true_d=true_d))
                    course_topics[course_id] = internal
    return courses, topics, offerings, course_topics, course_exams


# =============================================================================
# 6. STUDENTS AND REVIEW SIMULATION
# =============================================================================
def build_students(rng, cfg, institutions):
    students, n = [], 0
    for inst in institutions:
        college = inst["_level"] == "College"
        for _ in range(cfg.students_per_institution):
            n += 1
            students.append(dict(
                student_id=f"STU-{n:06d}", institution_id=inst["institution_id"], college=college,
                ability=float(rng.normal(0.1 if college else 0.0, 0.9 if college else 1.0)),
                engagement=float(rng.beta(2.2, 2.8)), cram=float(rng.beta(2.5, 2.5)),
                hour_mean=float(rng.normal(20.0 if college else 18.5, 2.0))))
    return students


def simulate_student_topic(rng, cfg, stu, tp, exam_dates, term):
    d, a, eng = tp["true_d"], stu["ability"], stu["engagement"]
    if rng.random() > float(np.clip(0.15 + 0.80 * eng, 0.05, 0.97)):
        return []                                           # student never reviewed this topic

    k0 = float(np.clip(0.55 + 0.17 * a - 0.30 * (d - 0.5) + rng.normal(0, 0.07), 0.05, 0.98))
    assess = None if rng.random() < 0.08 else round(float(np.clip(100 * (k0 + rng.normal(0, 0.06)), 15, 100)), 1)

    instr, final_day = tp["instruction_date"], exam_dates[-1]
    assessed = next((e for e in exam_dates if e > instr), final_day)
    n_rev = min(8, 1 + int(rng.poisson(0.7 + 3.2 * eng)))

    # ---- choose review dates: spaced reviewers vs. crammers ----
    cand, prev = [], instr
    for j in range(n_rev):
        if rng.random() < stu["cram"]:
            dd = assessed - timedelta(days=int(rng.integers(1, 6)))
        else:
            base = SPACING_BASE_DAYS[min(j, len(SPACING_BASE_DAYS) - 1)]
            dd = prev + timedelta(days=max(1, int(round(base * rng.lognormal(0, 0.35)))))
        prev = dd
        dd = max(dd, instr + timedelta(days=1))
        if dd >= final_day:
            continue
        if dd in term["_break_dates"] and rng.random() < 0.7:
            continue
        if dd > assessed and assessed != final_day and rng.random() < 0.5:
            continue
        cand.append(dd)
    dates = sorted(set(cand))
    if not dates:
        return []

    # ---- memory simulation: R = exp(-t / S) ----
    S = (BASE_STABILITY_DAYS * math.exp(0.30 * a) * (1.55 - d) * ((tp["instruction_minutes"] / 60) ** 0.25)
         * math.exp(2.0 * (k0 - 0.6)) * rng.lognormal(0, 0.25))
    last_mem_ts = datetime.combine(instr, time(11, 0))
    prev_ts, rows = None, []
    for dd in dates:
        hour = int(np.clip(rng.normal(stu["hour_mean"], 2.0), 6, 23))
        ts = datetime.combine(dd, time(hour, int(rng.integers(0, 60)), int(rng.integers(0, 60))))
        act = str(rng.choice(ACTIVITIES, p=ACTIVITY_P))
        lo, hi = ACTIVITY_ITEMS[act]
        n = int(rng.integers(lo, hi))
        R = math.exp(-((ts - last_mem_ts).total_seconds() / 86400) / S)

        if rng.random() < cfg.abandoned_session_rate:       # opened the tool, answered nothing
            n_att, correct, hints, rt, success = 0, 0, 0, None, None
        else:
            p = float(np.clip(GUESS_FLOOR + (1 - GUESS_FLOOR) * R, 0.02, 0.985))
            p_item = 1 / (1 + math.exp(-(math.log(p / (1 - p)) + rng.normal(0, 0.35))))
            correct0 = int(rng.binomial(n, p_item))          # unassisted correct answers
            hints = min(n, int(rng.poisson(0.10 + 1.6 * (1 - R) * (0.4 + d))))
            assisted = int(rng.binomial(min(hints, n - correct0), 0.7))   # hints turn some misses into "correct"
            correct, n_att = correct0 + assisted, n
            success = int(correct / n >= cfg.recall_threshold and hints <= cfg.max_hints_for_success)
            rt = None if rng.random() < 0.015 else round(float(rng.lognormal(math.log(5 + 18 * (1 - R) * (0.5 + d)), 0.3)), 1)

            q = correct0 / n                                  # stability update (spacing + testing effect)
            gain = ACTIVITY_GAIN[act]
            if q >= 0.8:
                S *= 1 + gain * (0.9 + 2.4 * (1 - R)) * (1 - 0.45 * d) * rng.lognormal(0, 0.15)
            elif q >= 0.5:
                S *= 1 + 0.15 * gain
            else:
                S = max(0.7, S * 0.55)
            last_mem_ts = ts

        rows.append(dict(
            student_id=stu["student_id"], institution_id=tp["institution_id"], course_id=tp["course_id"],
            topic_id=tp["topic_id"], term_id=tp["term_id"], learning_date=instr.isoformat(),
            review_timestamp=ts, previous_review_timestamp=prev_ts,
            days_since_last_review=None if prev_ts is None else int((ts - prev_ts).total_seconds() // 86400),
            activity_type=act, previous_assessment_score_pct=assess, items_attempted=n_att,
            items_correct=correct, hints_used=hints, response_time_seconds=rt, recall_success=success))
        prev_ts = ts
    return rows


def simulate_reviews(rng, cfg, students, terms_by_inst, offerings, course_topics, course_exams):
    rows = []
    for stu in students:
        k_courses = 4 if stu["college"] else int(rng.choice([4, 5]))
        for term in terms_by_inst[stu["institution_id"]]:
            codes = list(offerings[(stu["institution_id"], term["term_id"])])
            for code in rng.choice(codes, size=min(k_courses, len(codes)), replace=False):
                sections = offerings[(stu["institution_id"], term["term_id"])][str(code)]
                cid = sections[int(rng.integers(len(sections)))]
                for tp in course_topics[cid]:
                    rows.extend(simulate_student_topic(rng, cfg, stu, tp, course_exams[cid], term))
    df = pd.DataFrame(rows).sort_values("review_timestamp", kind="mergesort").reset_index(drop=True)
    df.insert(6, "review_id", [f"R{i + 1:08d}" for i in range(len(df))])
    return df


# =============================================================================
# 7. VALIDATION
# =============================================================================
def validate(cfg, inst, term, course, topic, ret):
    errs = []

    def check(cond, msg):
        if not bool(cond):
            errs.append(msg)

    for df, key, name in [(inst, ["institution_id"], "institution"), (term, ["term_id"], "term"),
                          (course, ["course_id"], "course"), (topic, ["topic_id"], "topic"),
                          (ret, ["review_id"], "student_retention")]:
        check(not df.duplicated(key).any(), f"duplicate primary key in {name}")

    def fk(child, ccols, parent, pcols, name):
        keys = set(map(tuple, parent[pcols].to_numpy()))
        check(all(tuple(r) in keys for r in child[ccols].drop_duplicates().to_numpy()), f"foreign key failed: {name}")

    fk(term, ["institution_id"], inst, ["institution_id"], "term->institution")
    fk(course, ["institution_id"], inst, ["institution_id"], "course->institution")
    fk(topic, ["course_id"], course, ["course_id"], "topic->course")
    fk(ret, ["topic_id", "course_id"], topic, ["topic_id", "course_id"], "retention->topic")
    fk(ret, ["term_id"], term, ["term_id"], "retention->term")
    fk(ret, ["institution_id"], inst, ["institution_id"], "retention->institution")

    check(course.exam_dates.notna().all(), "every course needs exam_dates")
    check((term.term_start_date < term.instruction_start_date).all(), "term start must precede instruction start")
    check((term.instruction_start_date <= term.instruction_end_date).all(), "instruction dates out of order")
    check((term.instruction_end_date < term.finals_start_date).all(), "instruction must end before finals")
    check((term.finals_start_date <= term.finals_end_date).all() and (term.finals_end_date <= term.term_end_date).all(),
          "finals/term end out of order")

    ts = pd.to_datetime(ret.review_timestamp)
    prev = pd.to_datetime(ret.previous_review_timestamp)
    has_prev = prev.notna()
    check((ts[has_prev] > prev[has_prev]).all(), "review not after previous review")
    check((ret.days_since_last_review.dropna() >= 0).all(), "negative interval")
    first = ~ret.duplicated(["student_id", "topic_id"], keep="first")   # rows are time-sorted globally
    check(first.eq(~has_prev).all(), "previous_review_timestamp should be null only for the first review")
    check((ret.items_correct <= ret.items_attempted).all(), "items_correct > items_attempted")
    check((ret.hints_used <= ret.items_attempted).all(), "hints_used > items_attempted")
    check(ret.previous_assessment_score_pct.dropna().between(0, 100).all(), "assessment score outside 0-100")
    check(ret.recall_success.isna().eq(ret.items_attempted == 0).all(), "recall_success null iff items_attempted == 0")
    lab = ((ret.items_correct / ret.items_attempted.replace(0, np.nan) >= cfg.recall_threshold) &
           (ret.hints_used <= cfg.max_hints_for_success)).astype(float).where(ret.items_attempted > 0)
    check(lab.fillna(-1).eq(ret.recall_success.astype(float).fillna(-1)).all(), "recall_success does not follow the documented rule")
    check(topic.empirical_difficulty.between(0, 1).all(), "empirical_difficulty outside 0-1")

    chk = ret.merge(term[["term_id", "instruction_start_date", "finals_end_date"]], on="term_id")
    d = pd.to_datetime(chk.review_timestamp).dt.normalize()
    check((d >= pd.to_datetime(chk.instruction_start_date)).all() and (d <= pd.to_datetime(chk.finals_end_date)).all(),
          "review outside its term")
    check((d >= pd.to_datetime(chk.learning_date)).all(), "review before the topic was taught")

    if errs:
        raise AssertionError("Validation failed:\n - " + "\n - ".join(errs))
    print("Validation: keys, chronology, ranges and label rule all passed.")


# =============================================================================
# 8. MAIN
# =============================================================================
def iso(df, cols):
    for c in cols:
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
    return df


def main():
    ap = argparse.ArgumentParser(description="Generate the 5 synthetic source datasets (US schools/colleges).")
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--students-per-institution", type=int, default=Config.students_per_institution)
    ap.add_argument("--n-high-schools", type=int, default=Config.n_high_schools)
    ap.add_argument("--n-colleges", type=int, default=Config.n_colleges)
    ap.add_argument("--academic-year-start", type=int, default=Config.academic_year_start)
    ap.add_argument("--out-dir", default=Config.out_dir, help="default: ./data next to this script")
    args = ap.parse_args()
    cfg = Config(seed=args.seed, students_per_institution=args.students_per_institution,
                 n_high_schools=args.n_high_schools, n_colleges=args.n_colleges,
                 academic_year_start=args.academic_year_start, out_dir=args.out_dir)
    rng = np.random.default_rng(cfg.seed)

    institutions = build_institutions(rng, cfg)
    terms_by_inst = {i["institution_id"]: build_terms(rng, i, cfg.academic_year_start) for i in institutions}
    courses, topics, offerings, course_topics, course_exams = build_catalog(rng, cfg, institutions, terms_by_inst)
    students = build_students(rng, cfg, institutions)
    print(f"Simulating review events for {len(students):,} students ...")
    ret = simulate_reviews(rng, cfg, students, terms_by_inst, offerings, course_topics, course_exams)

    inst_df = pd.DataFrame(institutions)[["institution_id", "institution_name", "institution_type", "state_code",
                                          "locale", "term_system", "grade_range", "timezone"]]
    term_df = pd.DataFrame([t for ts_ in terms_by_inst.values() for t in ts_])[
        ["term_id", "institution_id", "term_name", "academic_year", "term_type", "term_start_date", "term_end_date",
         "instruction_start_date", "instruction_end_date", "finals_start_date", "finals_end_date", "break_periods"]]
    course_df = pd.DataFrame(courses)
    topic_df = pd.DataFrame(topics)
    validate(cfg, inst_df, term_df, course_df, topic_df, ret)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    term_df = iso(term_df, ["term_start_date", "term_end_date", "instruction_start_date", "instruction_end_date",
                            "finals_start_date", "finals_end_date"])
    # Keep only the fields requested for Dataset 5. Finals/break columns are internal helpers.
    term_df = term_df[["term_id", "institution_id", "term_name", "academic_year", "term_type",
                       "term_start_date", "term_end_date", "instruction_start_date", "instruction_end_date"]]
    topic_df = iso(topic_df, ["instruction_date"])
    for c in ["review_timestamp", "previous_review_timestamp"]:
        ret[c] = pd.to_datetime(ret[c]).dt.strftime("%Y-%m-%d %H:%M:%S")      # local institutional time
    ret["recall_success"] = ret["recall_success"].astype("Int64")
    ret["days_since_last_review"] = ret["days_since_last_review"].astype("Int64")
    # Dataset 1: exactly the user-approved columns. review_id and hints_used are not exported.
    ret = ret[["student_id", "institution_id", "course_id", "topic_id", "term_id", "learning_date",
               "review_timestamp", "previous_review_timestamp", "days_since_last_review", "activity_type",
               "previous_assessment_score_pct", "items_attempted", "items_correct",
               "response_time_seconds", "recall_success"]]

    # Dataset 2: remove the simulation-only exam_dates field.
    course_df = course_df[["course_id", "institution_id", "course_title", "subject_area", "course_level",
                           "course_type", "credit_value", "modality", "instructor_id"]]

    files = {"student_retention.csv": ret, "course.csv": course_df, "topic.csv": topic_df,
             "institution.csv": inst_df, "term.csv": term_df}
    for name, df in files.items():
        df.to_csv(out / name, index=False, encoding="utf-8")
        print(f"  saved {out.name}/{name:22s} {len(df):>9,} rows x {df.shape[1]:>2} cols")
    print(f"\nDone. Next step: python dataset_processing.py   (reads {out.resolve()})")


if __name__ == "__main__":
    main()
