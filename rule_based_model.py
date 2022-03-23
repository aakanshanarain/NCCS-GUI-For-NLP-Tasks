"""
Moving toward front end html webpage
# just need the functions, no need to run anything, all commented out
"""

import pandas as pd
import re
import copy
import pprint
import os

path = os.path.dirname(__file__)

'''
# for flaskapp integration: put this into app.py

dataset_location = os.path.join(path, r"14122021 NLP Histo 3032 Reports (Classify Cancerous and Non-cancerous Reports).xlsx")
df_initial = pd.read_excel(dataset_location, usecols="D, F, H, P")
df_initial.rename(columns={"Grade(1, 2, 3, mildly or well = 1, moderately = 2, poorly = 3)": "grades"}, inplace=True)
df_initial['grades'] = df_initial['grades'].fillna(0) # this changes all NaN values in the grade column to 0
'''

###################################
####### ESSENTIAL FUNCTIONS #######
###################################

# STEP 1 FUNCTION
def convert_df(data):
    id_list = []
    text_list = []
    grades_list = []
    for index, row in data.iterrows():
      id = row['SCM GUIDE'].lower()
      text = (row['DIAGNOSIS'] + " " + row['MICROSCOPIC DESCRIPTION']).lower()
      grades = re.findall('\d+', str(row['grades']))
      # if there are no grades assigned, it is 'Unknown' --> treat as '0' (no grade)
      if len(grades) == 0:
        grades = ['0']
      id_list.append(id)
      text_list.append(text)
      grades_list.append(grades)
    converted_df = pd.DataFrame({'id': id_list, 'text': text_list, 'grades': grades_list})
    return converted_df


# STEP 2 FUNCTION
def find_matches(data):
    matches_list = []
    for index, row in data.iterrows():
      pattern = re.compile(r"[\s\S]{25}grade[\s\S]{25}|[\s\S]{25}differentiated[\s\S]{25}")
      matches = pattern.findall(row['text'])
      matches_list.append(matches) 
    data["matches"] = matches_list
    return data


# STEP 3 FUNCTION
def determine_grade(data):

    # for id in determined_data.keys():
    #   determined_data[id]["matches"] = None    # set "matches" = None for all rows in determined_data first

    determined_list = []

    # Edit patterns for the respective grades here
    pattern1 = re.compile(r"low|grade\s1|grade\si|well")
    pattern2 = re.compile(r"intermediate|grade\s2|grade\sii|moderately")
    pattern3 = re.compile(r"high|grade\s3|grade\siii|poorly")
    pattern_false = re.compile(r"dcis|dysplasia") # removed 'nuclear' for now.
    
    for index, row in data.iterrows():
      grades_list = []
      for match_text in row["matches"]:  
        grade1_match = pattern1.findall(match_text)
        grade2_match = pattern2.findall(match_text)
        grade3_match = pattern3.findall(match_text)
        matches_false = pattern_false.findall(match_text)

        if not matches_false:
          if grade1_match:
            grades_list.append('1')
          if grade2_match:
            grades_list.append('2')
          if grade3_match:
            grades_list.append('3')
          
        # grades_list.append('9')

      # If all matches in a given row does not result in any determined grade, set 'determined' = ['0']
      if len(grades_list) == 0:
        grades_list.append('0')
      determined_list.append(grades_list)
    data["determined"] = determined_list
      
    #   determined_data[id]["determined_grades"] = determined_grades
    #   determined_data[id]["matches"] = all_matches[id]
    return data


# STEP 4 FUNCTION
def evaluate_accuracy(data):
    correct_counter = 0
    total = len(data)
    result_list = []  # holds a string "Correct" or "Wrong" for each row after comparing "grades" with "determined"

    for index, row in data.iterrows():
      grades = row["grades"]
      determined_grades = row["determined"]

      if determined_grades == grades: # if determined grades matches actual grades exacty, it is correct
        correct_counter += 1
        result = "Correct"
      else:
        result = "Wrong"
      
      result_list.append(result)
    
    data["result"] = result_list

    score = correct_counter / total
    
    return data, score



###################################
##### MISCELLANEOUS FUNCTIONS #####
###################################

def wrong_gradings(data):
  id_list = []
  text_list = []
  grades_list = []
  matches_list = []
  determined_list = []
  result_list = []
  for index, row in data.iterrows():
    if row['result'] == 'Wrong':
      id_list.append(row['id'])
      text_list.append(row['text'])
      grades_list.append(row['grades'])
      matches_list.append(row['matches'])
      determined_list.append(row['determined'])
      result_list.append(row['result'])
    wrong_df = pd.DataFrame({'id': id_list, 'text': text_list, 'grades': grades_list, 'matches': matches_list, 'determined': determined_list, 'result': result_list})
  return wrong_df


def correct_gradings(data):
  id_list = []
  text_list = []
  grades_list = []
  matches_list = []
  determined_list = []
  result_list = []
  for index, row in data.iterrows():
    if row['result'] == 'Correct':
      id_list.append(row['id'])
      text_list.append(row['text'])
      grades_list.append(row['grades'])
      matches_list.append(row['matches'])
      determined_list.append(row['determined'])
      result_list.append(row['result'])
    correct_df = pd.DataFrame({'id': id_list, 'text': text_list, 'grades': grades_list, 'matches': matches_list, 'determined': determined_list, 'result': result_list})
  return correct_df


def false_positives(data):
    id_list = []
    text_list = []
    grades_list = []
    matches_list = []
    determined_list = []
    result_list = []
    for index, row in data.iterrows():
      # if there are detected matches, yet there are not supposed to be any grades at all (grades = ['0'])
      if len(row['matches']) != 0 and row['grades'] == ['0']:
        id_list.append(row['id'])
        text_list.append(row['text'])
        grades_list.append(row['grades'])
        matches_list.append(row['matches'])
        determined_list.append(row['determined'])
        result_list.append(row['result'])
      false_positives_df = pd.DataFrame({'id': id_list, 'text': text_list, 'grades': grades_list, 'matches': matches_list, 'determined': determined_list, 'result': result_list})
    return false_positives_df



def false_negatives(data):
    id_list = []
    text_list = []
    grades_list = []
    matches_list = []
    determined_list = []
    result_list = []
    for index, row in data.iterrows():
      # if there are NO detected matches, yet there are supposed to be some grade(s) assigned to the report.
      if len(row['matches']) == 0 and row['grades'] != ['0']:
        id_list.append(row['id'])
        text_list.append(row['text'])
        grades_list.append(row['grades'])
        matches_list.append(row['matches'])
        determined_list.append(row['determined'])
        result_list.append(row['result'])
      false_negatives_df = pd.DataFrame({'id': id_list, 'text': text_list, 'grades': grades_list, 'matches': matches_list, 'determined': determined_list, 'result': result_list})
    return false_negatives_df

'''
# for flaskapp integration: don't need to run anything, will do in app.py


# This section of the code utlises all the essential functions defined above.

# The naming of varaibles will start with 'df' and then state the columns in the
# corresponding dataframe, separated by a '_' (e.g. df_id_text_grades_matches
# means that the dataframe has 4 columns: id, text, grades, matches)

# The steps that the dataframe goes through are as such:

# Step 1: Convert DF to show ID, TEXT and GRADES
df = convert_df(df_initial)

# Step 2: Find text matches to the word 'grade' and 'differentiated' and store in list (+ MATCHES)
df = find_matches(df)

# Step 3: Determine the list of grades from the list of matches (+ DETERMINED)
df = determine_grade(df)

# Step 4: Evaulate if determined grade is "Correct" or "Wrong" and calculate overall accuracy score (+ RESULT)
df, accuracy_score = evaluate_accuracy(df)

# RESULTANT DATAFRAME AFTER ALL STEPS (6 COLUMNS)
print("The total number of REPORTS is: " + str(len(df)))
print("\n")
print("The accuracy of determine_grade function is: " + str(accuracy_score))
print("\n")
df

# dataframe for wrong gradings
wrong_df = wrong_gradings(df)

print("The number of wrongly graded rows is: " + str(len(wrong_df)))
wrong_df

# dataframe for correct gradings
correct_df = correct_gradings(df)

print("The number of correctly graded reports is: " + str(len(correct_df)))
correct_df

# dataframe for false positives
false_positives_df = false_positives(df)

print("The number of false_positives is: " + str(len(false_positives_df)))
false_positives_df

# dataframe for false negatives
false_negatives_df = false_negatives(df)

print("The number of false_negatives is: " + str(len(false_negatives_df)))
false_negatives_df

# exporting dataframes

try: 
    os.mkdir(os.path.join(path,r'csvfiles')) # create directory to store csv files
except OSError as error: 
    print(error)  

df.to_csv(os.path.join(path, r'csvfiles\df.csv')) # export to to csv file
wrong_df.to_csv(os.path.join(path, r'csvfiles\wrong_df.csv')) # export to to csv file
correct_df.to_csv(os.path.join(path, r'csvfiles\correct_df.csv')) # export to to csv file
false_positives_df.to_csv(os.path.join(path, r'csvfiles\false_positives_df.csv')) # export to to csv file
false_negatives_df.to_csv(os.path.join(path, r'csvfiles\false_negatives_df.csv')) # export to to csv file
'''