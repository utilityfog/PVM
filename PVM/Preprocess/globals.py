# Excluding 'plan' from our list of categorical variables because there is linearity in its categories (1-99)
RANDHIE_CATEGORICAL_VARIABLES = ['coins', 'site', 'female', 'child', 'fchild', 'hlthg', 'hlthf', 'hlthp', 'tookphys', 'idp']
# Excluding 'plan' from list of numeric variables as well because it should not be standardized; it is a linear categorical variable
# Also exclusing 'zper' since we don't use it for prediction and only use it for collapsing
# 'black' turned out to be no binary (not just 0 or 1) so we have decided to make it numeric
RANDHIE_NUMERIC_VARIABLES = ['black', 'mhi', 'year', 'income', 'xage', 'educdec', 'time', 'outpdol', 'drugdol', 'suppdol', 'mentdol', 'inpdol', 'meddol', 'totadm', 'inpmis', 'mentvis', 'mdvis', 'notmdvis', 'num', 'disea', 'physlm', 'ghindx', 'mdeoff', 'pioff', 'lfam', 'lpi', 'logc', 'fmde', 'xghindx', 'lnum', 'lnmeddol', 'binexp']

# heart_attack_prediction.csv variables
HEART_CATEGORICAL_VARIABLES = ['Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption', 'Previous Heart Problems', 'Medication Use', 'Sex', 'Diet', 'Country']
HEART_NUMERIC_VARIABLES = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week', 'Stress Level', 'Sedentary Hours Per Day', 'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']