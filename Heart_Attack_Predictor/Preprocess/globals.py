# Excluding 'plan' from our list of categorical variables because there is linearity in its categories (1-99)
RANDHIE_CATEGORICAL_VARIABLES = ['site', 'female', 'child', 'fchild', 'hlthg', 'hlthf', 'hlthp', 'tookphys']
# Excluding 'plan' from list of numeric variables as well because it should not be standardized; it is a linear categorical variable
# Also exclusing 'zper' since we don't use it for prediction and only use it for collapsing
RANDHIE_NUMERIC_VARIABLES = ['black', 'mhi', 'coins', 'year', 'income', 'xage', 'educdec', 'time', 'outpdol', 'drugdol', 'suppdol', 'mentdol', 'inpdol', 'meddol', 'totadm', 'inpmis', 'mentvis', 'mdvis', 'notmdvis', 'num', 'disea', 'physlm', 'ghindx', 'mdeoff', 'pioff', 'lfam', 'lpi', 'idp', 'logc', 'fmde', 'xghindx', 'linc', 'lnum', 'lnmeddol', 'binexp']

# heart_attack_prediction.csv variables
HEART_CATEGORICAL_VARIABLES = []
HEART_NUMERIC_VARIABLES = []