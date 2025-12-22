#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_FILE="src/agent/multi_agent_edu_copilot.py"
if [ ! -f "$SCRIPT_FILE" ]; then
    echo -e "${RED}Error: Python script file $SCRIPT_FILE not found. Please check the path!${NC}"
    exit 1
fi

# ************************** Subcommand 1: Generate Learning Plan **************************
echo -e "\n${YELLOW}=== Starting to Generate Learning Plan ===${NC}"
python "$SCRIPT_FILE" plan \
    --subject "Computer Science" \
    --target "Master Python Basic Syntax" \
    --time-available "5 hours per week for 4 weeks" \
    --current-level beginner \
    --current-topic "Python Basic Syntax"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== Learning Plan Generated Successfully ===${NC}\n"
else
    echo -e "${RED}=== Failed to Generate Learning Plan ===${NC}\n"
fi

# ************************** Subcommand 2: Generate Learning Materials **************************
echo -e "${YELLOW}=== Starting to Generate Learning Materials ===${NC}"
python "$SCRIPT_FILE" material \
    --subject "Computer Science" \
    --topic "Python Functions" \
    --current-level beginner
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== Learning Materials Generated Successfully ===${NC}\n"
else
    echo -e "${RED}=== Failed to Generate Learning Materials ===${NC}\n"
fi

# ************************** Subcommand 3: Evaluate Assignment **************************
echo -e "${YELLOW}=== Starting Assignment Evaluation ===${NC}"
# Note: Handle multi-line strings in Bash using single quotes to preserve line breaks
STUDENT_ANSWER='def average(numbers):
    return sum(numbers) / len(numbers)'
REFERENCE_ANSWER='def calculate_average(data_list):
    total = sum(data_list)
    count = len(data_list)
    return total / count if count != 0 else 0'

python "$SCRIPT_FILE" assess \
    --subject "Computer Science" \
    --question "Write a Python function to calculate the average of a list" \
    --student-answer "$STUDENT_ANSWER" \
    --reference-answer "$REFERENCE_ANSWER"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== Assignment Evaluation Completed ===${NC}\n"
else
    echo -e "${RED}=== Failed to Evaluate Assignment ===${NC}\n"
fi

# ************************** Subcommand 4: Explain Concept **************************
echo -e "${YELLOW}=== Starting Concept Explanation ===${NC}"
python "$SCRIPT_FILE" explain \
    --concept "Recursion in Functions" \
    --confusion-point "Don't understand how recursion terminates" \
    --current-level beginner
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== Concept Explanation Completed ===${NC}\n"
else
    echo -e "${RED}=== Failed to Explain Concept ===${NC}\n"
fi

echo -e "${YELLOW}All commands executed!${NC}"