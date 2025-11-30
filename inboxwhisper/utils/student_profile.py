"""
Shared profile information for the student the assistant supports.
"""

STUDENT_NAME = "pranav gupta"
STUDENT_ROLL = "2410110241"

def normalized_name():
    return STUDENT_NAME.lower().strip()

def normalized_roll():
    return STUDENT_ROLL.lower().strip()

