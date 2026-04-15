"""
ESS Variable Selection Schema for the Behavioral Grounding Framework.

This module defines which ESS Round 11 variables are used for population
synthesis, agent profiling, and behavioral modeling. It also defines
cleaning rules for ESS special codes.

References:
    - ESS11 Main Data (ESS11MD_e01_2.csv) — 865 respondents × 9,348 variables
    - ESS11 Interview Data (ESS11INTe04_1.csv) — interview metadata (24 variables)
    - ESS Data Documentation: https://www.europeansocialsurvey.org/methodology/
"""

# ---------------------------------------------------------------------------
# ESS missing-value conventions
# Values that encode "not applicable", "refusal", "don't know", "no answer"
# and should be treated as NaN during ingestion.
# ---------------------------------------------------------------------------
ESS_MISSING_CODES = {
    6,
    66,
    666,
    6666,
    66666,
    666666,  # not applicable
    7,
    77,
    777,
    7777,
    77777,
    777777,  # refusal
    8,
    88,
    888,
    8888,
    88888,
    888888,  # don't know
    9,
    99,
    999,
    9999,
    99999,
    999999,  # no answer
}

# For some variables, 55 = "not applicable"
ESS_MISSING_55 = {55, 555, 5555, 55555, 555555}


# ---------------------------------------------------------------------------
# Variable groups — each entry is (ess_column, target_name, description)
# ---------------------------------------------------------------------------

DEMOGRAPHICS = [
    ("cntry", "country", "Country"),
    ("gndr", "gender", "Gender (1=Male, 2=Female)"),
    ("agea", "age", "Age of respondent"),
    ("eisced", "education_level", "Highest level of education (ES-ISCED)"),
    ("hinctnta", "income_decile", "Household total net income decile"),
    ("emplrel", "employment_relation", "Employment relation"),
    ("isco08", "occupation_code", "Occupation (ISCO-08)"),
    ("domicil", "urbanization", "Domicile type (1=big city to 5=countryside)"),
    ("brncntr", "born_in_country", "Born in country (1=yes, 2=no)"),
    ("rlgblg", "religious_belonging", "Belonging to a religion (1=yes, 2=no)"),
]

TRUST = [
    ("ppltrst", "trust_people", "Most people can be trusted (0-10)"),
    ("pplfair", "trust_fairness", "Most people try to be fair (0-10)"),
    ("pplhlp", "trust_helpfulness", "Most people are helpful (0-10)"),
    ("trstprl", "trust_parliament", "Trust in parliament (0-10)"),
    ("trstlgl", "trust_legal", "Trust in the legal system (0-10)"),
    ("trstplc", "trust_police", "Trust in the police (0-10)"),
    ("trstplt", "trust_politicians", "Trust in politicians (0-10)"),
    ("trstprt", "trust_parties", "Trust in political parties (0-10)"),
    ("trstep", "trust_eu_parliament", "Trust in the European Parliament (0-10)"),
    ("trstun", "trust_un", "Trust in the United Nations (0-10)"),
]

POLITICS = [
    ("polintr", "political_interest", "Interest in politics (1=very to 4=not at all)"),
    ("lrscale", "left_right", "Left-right placement (0=left, 10=right)"),
    ("vote", "voted_last_election", "Voted in last national election (1=yes, 2=no, 3=not eligible)"),
    ("contplt", "contacted_politician", "Contacted politician (1=yes, 2=no)"),
    ("psppsgva", "political_self_efficacy", "Political system allows influence (1-5)"),
]

VALUES_ATTITUDES = [
    ("stflife", "life_satisfaction", "Life satisfaction (0-10)"),
    ("stfeco", "satisfaction_economy", "Satisfaction with economy (0-10)"),
    ("stfgov", "satisfaction_government", "Satisfaction with government (0-10)"),
    ("stfdem", "satisfaction_democracy", "Satisfaction with democracy (0-10)"),
    ("stfedu", "satisfaction_education", "Satisfaction with education system (0-10)"),
    ("stfhlth", "satisfaction_health_sys", "Satisfaction with health services (0-10)"),
    ("happy", "happiness", "How happy are you (0-10)"),
    ("gincdif", "reduce_inequality", "Government should reduce income differences (1-5)"),
    ("freehms", "gay_rights", "Gays and lesbians should be free to live (1-5)"),
    ("imsmetn", "immigration_same_ethnicity", "Allow immigrants same ethnicity (1-4)"),
    ("imdfetn", "immigration_diff_ethnicity", "Allow immigrants different ethnicity (1-4)"),
    ("impcntr", "immigration_poor_countries", "Allow immigrants from poorer countries (1-4)"),
]

SOCIAL = [
    ("sclmeet", "social_meeting_freq", "How often socially meet (1-7)"),
    ("inprdsc", "close_confidants", "Anyone to discuss intimate matters (1-7)"),
    ("sclact", "social_activity", "Take part in social activities (1-5)"),
    ("volunfp", "volunteered", "Worked in voluntary organisation last 12mo (1=yes, 2=no)"),
]

HEALTH_WELLBEING = [
    ("health", "self_rated_health", "Subjective general health (1=very good to 5=very bad)"),
    ("hlthhmp", "health_hampered", "Hampered by illness/disability (1=yes a lot to 3=no)"),
    ("fltdpr", "felt_depressed", "Felt depressed last week (1-4)"),
    ("flteeff", "felt_everything_effort", "Felt everything an effort (1-4)"),
    ("slprl", "sleep_restless", "Sleep was restless last week (1-4)"),
    ("wrhpp", "was_happy", "Were happy last week (1-4)"),
    ("fltlnl", "felt_lonely", "Felt lonely last week (1-4)"),
    ("enjlf", "enjoyed_life", "Enjoyed life last week (1-4)"),
    ("fltsd", "felt_sad", "Felt sad last week (1-4)"),
    ("cldgng", "could_not_get_going", "Could not get going last week (1-4)"),
]

RISK_PERSONALITY = [
    ("likrisk", "risk_taking", "To what extent like taking risks (0-6)"),
    ("liklead", "leadership_preference", "To what extent like to lead (0-6)"),
    ("actcomp", "competitiveness", "To what extent act competitively (0-6)"),
]

CRIME_SAFETY = [
    ("crmvct", "crime_victim", "Victim of crime last 5 years (1=yes, 2=no)"),
    ("aesfdrk", "feel_safe_dark", "Feel safe walking alone after dark (1-4)"),
]

# Interview metadata (from ESS11INTe04_1.csv)
INTERVIEW_META = [
    ("idno", "respondent_id", "Respondent ID"),
    ("cntry", "country", "Country"),
    ("intagea", "interviewer_age", "Interviewer age"),
    ("intgndr", "interviewer_gender", "Interviewer gender"),
    ("intlnga", "interview_language", "Interview language"),
]


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

ALL_VARIABLE_GROUPS = {
    "demographics": DEMOGRAPHICS,
    "trust": TRUST,
    "politics": POLITICS,
    "values_attitudes": VALUES_ATTITUDES,
    "social": SOCIAL,
    "health_wellbeing": HEALTH_WELLBEING,
    "risk_personality": RISK_PERSONALITY,
    "crime_safety": CRIME_SAFETY,
}


def get_ess_columns() -> list[str]:
    """Return all ESS column names to select from the main dataset."""
    cols = ["idno"]  # always need the join key
    for group in ALL_VARIABLE_GROUPS.values():
        for ess_col, _, _ in group:
            if ess_col not in cols:
                cols.append(ess_col)
    return cols


def get_rename_mapping() -> dict[str, str]:
    """Return ESS column → target name mapping."""
    mapping = {"idno": "respondent_id"}
    for group in ALL_VARIABLE_GROUPS.values():
        for ess_col, target, _ in group:
            mapping[ess_col] = target
    return mapping


def get_all_target_names() -> list[str]:
    """Return all target column names after renaming."""
    names = ["respondent_id"]
    for group in ALL_VARIABLE_GROUPS.values():
        for _, target, _ in group:
            if target not in names:
                names.append(target)
    return names
