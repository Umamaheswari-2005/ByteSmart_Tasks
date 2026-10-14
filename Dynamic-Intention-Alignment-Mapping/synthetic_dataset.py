"""
DIAM Synthetic Dataset Generator
---------------------------------
Generates realistic, linked, synthetic U.S.-college-style data for
Dynamic Intention Alignment Mapping (DIAM).

Raw tables:
    student_profile.csv
    intention_history.csv
    course_catalog.csv
    program_requirements.csv
    course_enrollment.csv
    academic_progress.csv
    advising_history.csv
    hidden_scenarios.csv

Important:
- This is synthetic data, not real institutional/IPEDS data.
- intention_history.csv records whether each intention is Stated or Inferred
  (intention_type), matching DIAM's "stated or inferred objectives".
- No target-like fields such as intention_changed or
  counts_toward_intended_program are written to raw tables.
- Data ends at 2026-Spring so 2026-Fall does not contain final grades.
"""

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_STUDENTS = 2000

TERMS = [
    "2023-Fall",
    "2024-Spring",
    "2024-Fall",
    "2025-Spring",
    "2025-Fall",
    "2026-Spring",
]

PROGRAMS = {
    "CERT-IT": {
        "program_name": "Information Technology Certificate",
        "degree_level": "Certificate",
        "credits_required": 30,
        "major_area": "IT",
    },
    "AS-IT": {
        "program_name": "Associate of Science in Information Technology",
        "degree_level": "Associate",
        "credits_required": 60,
        "major_area": "IT",
    },
    "BS-IT": {
        "program_name": "Bachelor of Science in Information Technology",
        "degree_level": "Bachelor",
        "credits_required": 120,
        "major_area": "IT",
    },
    "BS-CS": {
        "program_name": "Bachelor of Science in Computer Science",
        "degree_level": "Bachelor",
        "credits_required": 120,
        "major_area": "CS",
    },
    "BS-DS": {
        "program_name": "Bachelor of Science in Data Science",
        "degree_level": "Bachelor",
        "credits_required": 120,
        "major_area": "DS",
    },
    "BA-BUS": {
        "program_name": "Bachelor of Arts in Business Administration",
        "degree_level": "Bachelor",
        "credits_required": 120,
        "major_area": "BUS",
    },
    "AS-BUS": {
        "program_name": "Associate of Science in Business",
        "degree_level": "Associate",
        "credits_required": 60,
        "major_area": "BUS",
    },
}

# Synthetic enrollment assumptions. They are intentionally configurable and
# should not be presented as official IPEDS/NCES estimates.
FULL_TIME_PROB = {
    "Certificate": 0.55,
    "Associate": 0.63,
    "Bachelor": 0.78,
}

SCENARIO_PROBS = {
    "stable_alignment": 0.40,
    "successful_intention_change": 0.20,
    "misaligned_intention_change": 0.20,
    "academic_difficulty": 0.10,
    "program_change": 0.10,
}

# DIAM tracks stated and inferred educational objectives. Student-declared
# sources are "Stated"; sources derived from records/behaviour are "Inferred".
INTENTION_TYPE = {
    "Student Survey": "Stated",
    "Academic Advisor": "Stated",
    "Degree Audit": "Inferred",
    "Career Planning": "Inferred",
}

STOP_OUT_PROB = {
    "stable_alignment": 0.025,
    "successful_intention_change": 0.040,
    "misaligned_intention_change": 0.085,
    "academic_difficulty": 0.120,
    "program_change": 0.050,
}


def default_data_dir() -> Path:
    """Folder of this script if it is already named 'data', else <script folder>/data."""
    here = Path(__file__).resolve().parent
    return here if here.name == "data" else here / "data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(default_data_dir()),
        help="Directory where CSV files will be written. Default: the data folder next to this script",
    )
    parser.add_argument(
        "--students",
        type=int,
        default=N_STUDENTS,
        help=f"Number of students. Default: {N_STUDENTS}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed. Default: {SEED}",
    )
    return parser.parse_args()


def term_index(term: str) -> int:
    return TERMS.index(term)


def add_terms(term: str, n_terms: int) -> str:
    idx = min(max(term_index(term) + n_terms, 0), len(TERMS) - 1)
    return TERMS[idx]


def class_level(credits: float) -> str:
    if credits < 30:
        return "Freshman"
    if credits < 60:
        return "Sophomore"
    if credits < 90:
        return "Junior"
    return "Senior"


def build_course_catalog(rng: random.Random) -> pd.DataFrame:
    courses = [
        # IT
        ("IT101", "Introduction to Information Technology", "IT", 3, 100, None),
        ("IT110", "Computer Applications", "IT", 3, 100, None),
        ("IT120", "Networking Fundamentals", "IT", 3, 100, None),
        ("IT201", "Systems Analysis", "IT", 3, 200, "IT101"),
        ("IT210", "Database Fundamentals", "IT", 3, 200, "IT101"),
        ("IT220", "Web Systems", "IT", 3, 200, "IT101"),
        ("IT301", "Cloud Computing", "IT", 3, 300, "IT120"),
        ("IT310", "Information Security", "IT", 3, 300, "IT120"),
        ("IT320", "IT Project Management", "IT", 3, 300, "IT201"),
        ("IT401", "Enterprise Systems", "IT", 3, 400, "IT201"),
        ("IT410", "Advanced Information Security", "IT", 3, 400, "IT310"),
        ("IT420", "IT Capstone", "IT", 3, 400, "IT320"),
        # CS
        ("CS101", "Programming I", "CS", 3, 100, None),
        ("CS120", "Computer Science Foundations", "CS", 3, 100, None),
        ("CS201", "Programming II", "CS", 3, 200, "CS101"),
        ("CS210", "Data Structures", "CS", 3, 200, "CS201"),
        ("CS220", "Computer Organization", "CS", 3, 200, "CS120"),
        ("CS301", "Algorithms", "CS", 3, 300, "CS210"),
        ("CS310", "Operating Systems", "CS", 3, 300, "CS220"),
        ("CS320", "Database Systems", "CS", 3, 300, "CS201"),
        ("CS401", "Software Engineering", "CS", 3, 400, "CS301"),
        ("CS410", "Computer Networks", "CS", 3, 400, "CS220"),
        ("CS420", "Computer Science Capstone", "CS", 3, 400, "CS401"),
        # DS
        ("DS101", "Introduction to Data Science", "DS", 3, 100, None),
        ("DS110", "Data Literacy", "DS", 3, 100, None),
        ("DS201", "Data Wrangling", "DS", 3, 200, "DS101"),
        ("DS210", "Statistics for Data Science", "DS", 3, 200, "MATH201"),
        ("DS220", "Python for Data Science", "DS", 3, 200, "CS101"),
        ("DS301", "Machine Learning", "DS", 3, 300, "DS201"),
        ("DS310", "Data Visualization", "DS", 3, 300, "DS201"),
        ("DS320", "Database Analytics", "DS", 3, 300, "DS201"),
        ("DS401", "Applied Predictive Modeling", "DS", 3, 400, "DS301"),
        ("DS410", "Data Science Capstone", "DS", 3, 400, "DS401"),
        # Business
        ("BUS101", "Introduction to Business", "BUS", 3, 100, None),
        ("BUS110", "Business Communication", "BUS", 3, 100, None),
        ("BUS201", "Financial Accounting", "BUS", 3, 200, "BUS101"),
        ("BUS210", "Managerial Accounting", "BUS", 3, 200, "BUS201"),
        ("BUS220", "Principles of Marketing", "BUS", 3, 200, "BUS101"),
        ("BUS301", "Operations Management", "BUS", 3, 300, "BUS101"),
        ("BUS310", "Business Analytics", "BUS", 3, 300, "MATH201"),
        ("BUS320", "Organizational Behavior", "BUS", 3, 300, "BUS101"),
        ("BUS401", "Strategic Management", "BUS", 3, 400, "BUS301"),
        ("BUS410", "Business Capstone", "BUS", 3, 400, "BUS401"),
        # General education
        ("ENG101", "College Writing", "GENED", 3, 100, None),
        ("ENG201", "Writing in the Disciplines", "GENED", 3, 200, "ENG101"),
        ("MATH101", "College Algebra", "GENED", 3, 100, None),
        ("MATH201", "Statistics", "GENED", 4, 200, "MATH101"),
        ("MATH210", "Applied Calculus", "GENED", 3, 200, "MATH101"),
        ("SCI101", "General Science", "GENED", 3, 100, None),
        ("SCI110", "Laboratory Science", "GENED", 4, 100, "SCI101"),
        ("HIST101", "U.S. History", "GENED", 3, 100, None),
        ("COMM101", "Public Speaking", "GENED", 3, 100, None),
        ("PSY101", "Introduction to Psychology", "GENED", 3, 100, None),
        ("SOC101", "Introduction to Sociology", "GENED", 3, 100, None),
        ("ART101", "Arts and Humanities", "GENED", 3, 100, None),
        ("ECON101", "Principles of Economics", "GENED", 3, 100, None),
        ("ELEC200", "General Elective Seminar", "ELECTIVE", 3, 200, None),
    ]

    rows = []
    for cid, name, area, credits, level, prereq in courses:
        rows.append(
            {
                "course_id": cid,
                "course_name": name,
                "subject_area": area,
                "credits": credits,
                "course_level": level,
                "prerequisite_course_id": prereq or "",
                "delivery_mode": rng.choices(
                    ["In Person", "Online", "Hybrid"],
                    weights=[0.55, 0.20, 0.25],
                    k=1,
                )[0],
            }
        )
    return pd.DataFrame(rows)


def build_program_requirements(course_catalog: pd.DataFrame) -> pd.DataFrame:
    course_info = course_catalog.set_index("course_id").to_dict("index")

    major_core = {
        "CERT-IT": ["IT101", "IT110", "IT120", "IT201", "IT210", "IT220"],
        "AS-IT": ["IT101", "IT110", "IT120", "IT201", "IT210", "IT220", "IT301", "IT310"],
        "BS-IT": ["IT101", "IT110", "IT120", "IT201", "IT210", "IT220", "IT301", "IT310", "IT320", "IT401", "IT410", "IT420"],
        "BS-CS": ["CS101", "CS120", "CS201", "CS210", "CS220", "CS301", "CS310", "CS320", "CS401", "CS410", "CS420"],
        "BS-DS": ["DS101", "DS110", "DS201", "DS210", "DS220", "DS301", "DS310", "DS320", "DS401", "DS410"],
        "AS-BUS": ["BUS101", "BUS110", "BUS201", "BUS220", "BUS301", "BUS320"],
        "BA-BUS": ["BUS101", "BUS110", "BUS201", "BUS210", "BUS220", "BUS301", "BUS310", "BUS320", "BUS401", "BUS410"],
    }

    # Fixed gen-ed package is intentionally synthetic but internally coherent.
    gened_by_level = {
        "Certificate": ["ENG101", "MATH101", "COMM101", "SCI101"],
        "Associate": ["ENG101", "ENG201", "MATH101", "MATH201", "SCI101", "HIST101", "COMM101", "PSY101"],
        "Bachelor": ["ENG101", "ENG201", "MATH101", "MATH201", "MATH210", "SCI101", "SCI110", "HIST101", "COMM101", "PSY101", "SOC101", "ART101"],
    }

    rows = []
    req_num = 1

    for program_id, program in PROGRAMS.items():
        core_ids = major_core[program_id]
        degree_level = program["degree_level"]

        core_credits = sum(course_info[c]["credits"] for c in core_ids)
        for idx, cid in enumerate(core_ids, start=1):
            c = course_info[cid]
            rows.append(
                {
                    "requirement_id": f"REQ-{req_num:05d}",
                    "program_id": program_id,
                    "requirement_type": "Major Core",
                    "course_group": "MAJOR_CORE",
                    "course_id": cid,
                    "required_credits": c["credits"],
                    "recommended_term": min(idx, 8),
                    "prerequisite": c["prerequisite_course_id"],
                    "is_gateway": int(idx <= 2 or c["course_level"] >= 300 and idx == len(core_ids)),
                }
            )
            req_num += 1

        gened_ids = gened_by_level[degree_level]
        gened_credits = sum(course_info[c]["credits"] for c in gened_ids)

        # Keep a modest fixed general-ed package and let the elective group
        # absorb the remaining credits. This guarantees exact program totals.
        for idx, cid in enumerate(gened_ids, start=1):
            c = course_info[cid]
            rows.append(
                {
                    "requirement_id": f"REQ-{req_num:05d}",
                    "program_id": program_id,
                    "requirement_type": "General Education",
                    "course_group": "GENERAL_ED",
                    "course_id": cid,
                    "required_credits": c["credits"],
                    "recommended_term": min(idx, 6),
                    "prerequisite": c["prerequisite_course_id"],
                    "is_gateway": int(cid in {"ENG101", "MATH101", "MATH201"}),
                }
            )
            req_num += 1

        fixed_credits = core_credits + gened_credits
        elective_credits = max(0, program["credits_required"] - fixed_credits)

        rows.append(
            {
                "requirement_id": f"REQ-{req_num:05d}",
                "program_id": program_id,
                "requirement_type": "Elective",
                "course_group": "FREE_ELECTIVE",
                "course_id": "",
                "required_credits": elective_credits,
                "recommended_term": 3,
                "prerequisite": "",
                "is_gateway": 0,
            }
        )
        req_num += 1

        total = fixed_credits + elective_credits
        assert total == program["credits_required"], (
            program_id,
            fixed_credits,
            elective_credits,
            total,
        )

    return pd.DataFrame(rows)


def choose_initial_program(rng: random.Random) -> str:
    return rng.choices(
        list(PROGRAMS),
        weights=[0.08, 0.18, 0.25, 0.20, 0.10, 0.14, 0.05],
        k=1,
    )[0]


def choose_upgrade_program(initial_program: str, rng: random.Random) -> str:
    upgrades = {
        "CERT-IT": ["BS-IT"],
        "AS-IT": ["BS-IT", "BS-CS"],
        "AS-BUS": ["BA-BUS"],
        "BS-IT": ["BS-CS", "BS-DS"],
        "BS-CS": ["BS-DS", "BS-IT"],
        "BS-DS": ["BS-CS", "BS-IT"],
        "BA-BUS": ["BS-IT", "BS-DS"],
    }
    return rng.choice(upgrades[initial_program])


def choose_lateral_program(initial_program: str, rng: random.Random) -> str:
    lateral = {
        "CERT-IT": ["AS-IT", "BS-IT"],
        "AS-IT": ["CERT-IT", "AS-BUS", "BS-IT"],
        "BS-IT": ["BS-CS", "BS-DS"],
        "BS-CS": ["BS-IT", "BS-DS"],
        "BS-DS": ["BS-CS", "BS-IT"],
        "BA-BUS": ["BS-IT", "BS-DS"],
        "AS-BUS": ["BA-BUS", "AS-IT"],
    }
    return rng.choice(lateral[initial_program])


def build_students(
    rng: random.Random,
    n_students: int,
    scenarios: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    student_rows = []
    hidden_rows = []

    for i in range(1, n_students + 1):
        sid = f"S{i:05d}"
        scenario = rng.choices(
            list(SCENARIO_PROBS),
            weights=list(SCENARIO_PROBS.values()),
            k=1,
        )[0]

        initial_program = choose_initial_program(rng)

        if scenario in {"successful_intention_change", "misaligned_intention_change"}:
            new_program = choose_upgrade_program(initial_program, rng)
            change_term_idx = rng.choice([2, 3, 4])
        elif scenario == "program_change":
            new_program = choose_lateral_program(initial_program, rng)
            change_term_idx = rng.choice([2, 3, 4])
        else:
            new_program = initial_program
            change_term_idx = None

        degree_level = PROGRAMS[initial_program]["degree_level"]
        transfer_student = rng.random() < 0.22

        if transfer_student:
            prior_credits = rng.choice([12, 15, 18, 24, 30, 36, 45, 60])
        else:
            prior_credits = 0

        first_time_student = int(not transfer_student)

        institution_type = rng.choices(
            ["Public 2-year", "Public 4-year", "Private nonprofit 4-year"],
            weights=[0.38, 0.52, 0.10],
            k=1,
        )[0]

        # More realistic part-time prevalence for certificates/associate programs.
        ft_prob = FULL_TIME_PROB[degree_level]
        if institution_type == "Public 2-year":
            ft_prob -= 0.06
        elif institution_type == "Private nonprofit 4-year":
            ft_prob += 0.04
        full_time = int(rng.random() < max(0.40, min(0.90, ft_prob)))

        entry_term = rng.choice([TERMS[0], TERMS[1], TERMS[2]])

        student_rows.append(
            {
                "student_id": sid,
                "entry_term": entry_term,
                "entry_program": initial_program,
                "entry_degree_level": degree_level,
                "major_area": PROGRAMS[initial_program]["major_area"],
                "first_time_student": first_time_student,
                "transfer_student": int(transfer_student),
                "prior_credits_accepted": prior_credits,
                "institution_type": institution_type,
                "typical_enrollment_status": "Full-Time" if full_time else "Part-Time",
            }
        )

        hidden_rows.append(
            {
                "student_id": sid,
                "scenario": scenario,
                "initial_program": initial_program,
                "change_term": TERMS[change_term_idx] if change_term_idx is not None else "",
                "new_program": new_program if change_term_idx is not None else "",
            }
        )

    return pd.DataFrame(student_rows), pd.DataFrame(hidden_rows)


def build_intention_history(
    students: pd.DataFrame,
    hidden: pd.DataFrame,
    rng: random.Random,
) -> pd.DataFrame:
    hidden_map = hidden.set_index("student_id").to_dict("index")
    rows = []

    for student in students.itertuples(index=False):
        h = hidden_map[student.student_id]
        change_term = h["change_term"]
        current_program = h["initial_program"]

        for term in TERMS:
            if term_index(term) < term_index(student.entry_term):
                continue

            if change_term and term_index(term) >= term_index(change_term):
                current_program = h["new_program"]

            # Same single RNG draw as before, so generated data is unchanged.
            source = rng.choices(
                ["Student Survey", "Academic Advisor", "Degree Audit", "Career Planning"],
                weights=[0.50, 0.18, 0.22, 0.10],
                k=1,
            )[0]

            rows.append(
                {
                    "student_id": student.student_id,
                    "term": term,
                    "intended_program": current_program,
                    "intention_source": source,
                    "intention_type": INTENTION_TYPE[source],
                }
            )

    return pd.DataFrame(rows)


def build_advising_history(
    students: pd.DataFrame,
    hidden: pd.DataFrame,
    rng: random.Random,
) -> pd.DataFrame:
    hidden_map = hidden.set_index("student_id").to_dict("index")
    rows = []
    appointment_id = 1

    for student in students.itertuples(index=False):
        h = hidden_map[student.student_id]
        change_idx = term_index(h["change_term"]) if h["change_term"] else None
        difficulty = h["scenario"] == "academic_difficulty"

        for term in TERMS:
            if term_index(term) < term_index(student.entry_term):
                continue

            idx = term_index(term)
            visits = 0.55

            # Advising activity rises before a change.
            if change_idx is not None and idx in {change_idx - 2, change_idx - 1}:
                visits += 0.90
            if change_idx is not None and idx == change_idx:
                visits += 0.60

            if difficulty:
                visits += 0.35

            count = min(3, rng.poisson(visits) if hasattr(rng, "poisson") else np.random.default_rng(rng.randint(0, 10**9)).poisson(visits))

            # Python Random has no poisson; the fallback above is deterministic
            # from this local seed source and keeps the dependency minimal.
            for _ in range(int(count)):
                after_change = change_idx is not None and idx >= change_idx

                if after_change and h["scenario"] == "successful_intention_change":
                    plan_updated = int(rng.random() < 0.72)
                    program_request = int(rng.random() < 0.60)
                elif after_change and h["scenario"] == "misaligned_intention_change":
                    plan_updated = int(rng.random() < 0.22)
                    program_request = int(rng.random() < 0.48)
                elif h["scenario"] == "program_change" and after_change:
                    plan_updated = int(rng.random() < 0.62)
                    program_request = int(rng.random() < 0.68)
                else:
                    plan_updated = int(rng.random() < 0.14)
                    program_request = 0

                rows.append(
                    {
                        "appointment_id": f"ADV-{appointment_id:07d}",
                        "student_id": student.student_id,
                        "term": term,
                        "advising_type": rng.choices(
                            ["Academic Planning", "Degree Audit", "Career Planning", "General Advising"],
                            weights=[0.36, 0.28, 0.18, 0.18],
                            k=1,
                        )[0],
                        "academic_plan_updated": plan_updated,
                        "program_change_requested": program_request,
                    }
                )
                appointment_id += 1

    return pd.DataFrame(rows)


def prerequisite_satisfied(
    course_id: str,
    passed_courses: set[str],
    course_info: dict,
) -> bool:
    prereq = course_info[course_id]["prerequisite_course_id"]
    return not prereq or prereq in passed_courses


def select_courses(
    target_program: str,
    passed_courses: set[str],
    attempted_courses: set[str],
    cumulative_credits: float,
    full_time: bool,
    course_info: dict,
    program_reqs: pd.DataFrame,
    term_number: int,
    rng: random.Random,
    scenario: str,
    difficulty_index: float,
) -> list[str]:
    reqs = program_reqs[program_reqs["program_id"] == target_program]

    required_course_ids = [
        x for x in reqs["course_id"].tolist() if x and x in course_info
    ]

    # A synthetic elective pool across areas. Electives do not need a target
    # course ID but can still contribute to degree progress.
    elective_pool = [
        "ART101",
        "ECON101",
        "PSY101",
        "SOC101",
        "COMM101",
        "ELEC200",
        "BUS110",
        "IT110",
        "CS120",
        "DS110",
    ]

    eligible = []
    for cid in required_course_ids:
        c = course_info[cid]
        if cid in passed_courses:
            continue
        if not prerequisite_satisfied(cid, passed_courses, course_info):
            continue
        # Avoid advanced courses too early unless the student has sufficient
        # credits to plausibly be at that level.
        if c["course_level"] >= 400 and cumulative_credits < 60:
            continue
        if c["course_level"] >= 300 and cumulative_credits < 30:
            continue
        eligible.append(cid)

    # Fill with electives only when there are not enough required courses.
    for cid in elective_pool:
        if cid not in course_info:
            continue
        if cid in passed_courses or cid in attempted_courses:
            continue
        if prerequisite_satisfied(cid, passed_courses, course_info):
            eligible.append(cid)

    # Difficulty makes students more likely to register for fewer courses.
    if full_time:
        target_load = rng.choice([12, 12, 15, 15, 15, 16])
    else:
        target_load = rng.choice([3, 6, 6, 9, 9])

    if scenario == "academic_difficulty":
        target_load = max(3, target_load - rng.choice([3, 3, 6]))
    elif scenario == "misaligned_intention_change" and term_number >= 3:
        target_load = max(3, target_load - rng.choice([0, 3]))

    rng.shuffle(eligible)

    selected = []
    credits = 0
    for cid in eligible:
        c = course_info[cid]
        if credits + c["credits"] <= target_load:
            selected.append(cid)
            credits += c["credits"]

    return selected


def build_enrollment_and_progress(
    students: pd.DataFrame,
    hidden: pd.DataFrame,
    intention: pd.DataFrame,
    course_catalog: pd.DataFrame,
    program_reqs: pd.DataFrame,
    rng: random.Random,
):
    course_info = course_catalog.set_index("course_id").to_dict("index")
    hidden_map = hidden.set_index("student_id").to_dict("index")
    intention_map = {
        (r.student_id, r.term): r.intended_program
        for r in intention.itertuples(index=False)
    }

    enrollment_rows = []
    progress_rows = []

    for student in students.itertuples(index=False):
        h = hidden_map[student.student_id]
        scenario = h["scenario"]

        passed_courses: set[str] = set()
        attempted_courses: set[str] = set()

        cumulative_earned = float(student.prior_credits_accepted)
        cumulative_attempted = float(student.prior_credits_accepted)
        cumulative_gpa_points = 0.0
        cumulative_gpa_credits = 0.0
        last_gpa = 3.0
        stop_out_previous = False

        start_idx = term_index(student.entry_term)

        for idx in range(start_idx, len(TERMS)):
            term = TERMS[idx]
            intended_program = intention_map[(student.student_id, term)]
            program = PROGRAMS[intended_program]

            full_time = (
                student.typical_enrollment_status == "Full-Time"
                and rng.random() < 0.94
            )

            # Stop-out probability rises with difficulty/misalignment.
            stopout_prob = STOP_OUT_PROB[scenario]
            if stop_out_previous:
                stopout_prob *= 0.55

            stop_out = rng.random() < stopout_prob

            registration_hold = int(
                rng.random()
                < (
                    0.035
                    + (0.035 if scenario == "academic_difficulty" else 0)
                    + (0.025 if stop_out else 0)
                )
            )

            selected_courses = []
            if not stop_out and not registration_hold:
                selected_courses = select_courses(
                    target_program=intended_program,
                    passed_courses=passed_courses,
                    attempted_courses=attempted_courses,
                    cumulative_credits=cumulative_earned,
                    full_time=full_time,
                    course_info=course_info,
                    program_reqs=program_reqs,
                    term_number=idx - start_idx + 1,
                    rng=rng,
                    scenario=scenario,
                    difficulty_index=idx - start_idx,
                )

            term_attempted = 0.0
            term_earned = 0.0
            term_quality_points = 0.0
            term_quality_credits = 0.0
            failed_courses = 0
            withdrawn_courses = 0

            # Academic difficulty is a term-level drift, not a per-course
            # permanent collapse. GPA can recover in later terms.
            if scenario == "academic_difficulty":
                difficulty_shift = -0.28 * max(0, idx - start_idx - 1)
                grade_mean = max(1.9, 3.0 + difficulty_shift)
            elif scenario == "misaligned_intention_change" and h["change_term"]:
                since_change = max(0, idx - term_index(h["change_term"]))
                grade_mean = max(2.25, 3.05 - 0.08 * since_change)
            else:
                grade_mean = min(3.35, 3.05 + 0.05 * min(idx - start_idx, 3))

            for cid in selected_courses:
                c = course_info[cid]
                credits = float(c["credits"])
                term_attempted += credits
                attempted_courses.add(cid)

                noise = rng.gauss(0, 0.48)
                grade_numeric = max(0.0, min(4.0, grade_mean + noise))

                if grade_numeric >= 3.55:
                    grade = "A"
                    earned = credits
                    points = 4.0
                    passed_courses.add(cid)
                elif grade_numeric >= 2.75:
                    grade = "B"
                    earned = credits
                    points = 3.0
                    passed_courses.add(cid)
                elif grade_numeric >= 1.95:
                    grade = "C"
                    earned = credits
                    points = 2.0
                    passed_courses.add(cid)
                elif grade_numeric >= 1.0:
                    grade = "D"
                    earned = credits
                    points = 1.0
                    passed_courses.add(cid)
                else:
                    # Some attempts fail. A later term can retake them because
                    # only passed courses are blocked from future enrollment.
                    if rng.random() < 0.30:
                        grade = "W"
                        earned = 0.0
                        points = 0.0
                        withdrawn_courses += 1
                    else:
                        grade = "F"
                        earned = 0.0
                        points = 0.0
                        failed_courses += 1

                term_earned += earned
                if grade != "W":
                    term_quality_points += points * credits
                    term_quality_credits += credits

                enrollment_rows.append(
                    {
                        "student_id": student.student_id,
                        "term": term,
                        "course_id": cid,
                        "credits_attempted": credits,
                        "final_grade": grade,
                        "credits_earned": earned,
                        "delivery_mode": c["delivery_mode"],
                    }
                )

            cumulative_earned += term_earned
            cumulative_attempted += term_attempted
            cumulative_gpa_points += term_quality_points
            cumulative_gpa_credits += term_quality_credits

            if term_quality_credits > 0:
                term_gpa = term_quality_points / term_quality_credits
                last_gpa = 0.65 * last_gpa + 0.35 * term_gpa
            else:
                term_gpa = last_gpa

            cumulative_gpa = (
                cumulative_gpa_points / cumulative_gpa_credits
                if cumulative_gpa_credits > 0
                else last_gpa
            )

            target_credits = program["credits_required"]
            remaining_credits = max(0.0, target_credits - cumulative_earned)

            expected_load = 15.0 if full_time else 9.0
            terms_needed = math.ceil(remaining_credits / expected_load) if remaining_credits else 0
            projected_graduation = add_terms(term, terms_needed)

            enrollment_status = "Stop-Out" if stop_out else (
                "Full-Time" if full_time else "Part-Time"
            )

            progress_rows.append(
                {
                    "student_id": student.student_id,
                    "term": term,
                    "intended_program": intended_program,
                    "enrollment_status": enrollment_status,
                    "registration_hold_flag": registration_hold,
                    "credits_attempted_term": term_attempted,
                    "credits_earned_term": term_earned,
                    "cumulative_credits_attempted": cumulative_attempted,
                    "cumulative_credits_earned": cumulative_earned,
                    "term_gpa": round(term_gpa, 3),
                    "cumulative_gpa": round(cumulative_gpa, 3),
                    "failed_courses_term": failed_courses,
                    "withdrawn_courses_term": withdrawn_courses,
                    "class_level": class_level(cumulative_earned),
                    "remaining_program_credits": remaining_credits,
                    "projected_graduation_term": projected_graduation,
                    "program_completion_flag": int(remaining_credits <= 0),
                }
            )

            stop_out_previous = stop_out

            # Once the program is complete, stop generating additional
            # coursework for that student.
            if remaining_credits <= 0:
                break

    return pd.DataFrame(enrollment_rows), pd.DataFrame(progress_rows)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = list(SCENARIO_PROBS)

    course_catalog = build_course_catalog(rng)
    program_requirements = build_program_requirements(course_catalog)

    students, hidden = build_students(
        rng=rng,
        n_students=args.students,
        scenarios=scenarios,
    )

    intention_history = build_intention_history(students, hidden, rng)
    advising_history = build_advising_history(students, hidden, rng)

    course_enrollment, academic_progress = build_enrollment_and_progress(
        students=students,
        hidden=hidden,
        intention=intention_history,
        course_catalog=course_catalog,
        program_reqs=program_requirements,
        rng=rng,
    )

    students.to_csv(output_dir / "student_profile.csv", index=False)
    intention_history.to_csv(output_dir / "intention_history.csv", index=False)
    course_catalog.to_csv(output_dir / "course_catalog.csv", index=False)
    program_requirements.to_csv(output_dir / "program_requirements.csv", index=False)
    course_enrollment.to_csv(output_dir / "course_enrollment.csv", index=False)
    academic_progress.to_csv(output_dir / "academic_progress.csv", index=False)
    advising_history.to_csv(output_dir / "advising_history.csv", index=False)

    # Debug/evaluation only. Do NOT feed this table to the model.
    hidden.to_csv(output_dir / "hidden_scenarios.csv", index=False)

    print(f"Generated synthetic DIAM data in: {output_dir}")
    for path in sorted(output_dir.glob("*.csv")):
        df = pd.read_csv(path)
        print(f"{path.name:35s} {len(df):>8,} rows")


if __name__ == "__main__":
    main()
