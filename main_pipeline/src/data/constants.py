# change to args from com line
DATA_PATH = '/Users/pashase_/Desktop/zillow-prize-1'

PROP_DATA_PATH = f'{DATA_PATH}/properties_2016.csv'
TRAIN_DATA_PATH = f'{DATA_PATH}/train_2016_v2.csv'
SAMPLE_SUB_DATA_PATH = f'{DATA_PATH}/sample_submission.csv'

PATH_TO_SAVE_CLEAN_DF = f'{DATA_PATH}/clean_df.csv'

TRAINED_MODEL_PATH = f'{DATA_PATH}/models/rf_model.sav'

SPLIT_TRAIN_DATA_PATH = f'{DATA_PATH}/split/train.csv'
SPLIT_TEST_DATA_PATH = f'{DATA_PATH}/split/test.csv'

# categorical features
CATEGORICAL_FEATURES_NAMES = [
    'airconditioningtypeid', 'architecturalstyletypeid', 'buildingqualitytypeid',
    'buildingclasstypeid', 'decktypeid', 'fireplaceflag',
    'hashottuborspa', 'heatingorsystemtypeid', 'pooltypeid10',
    'pooltypeid2', 'pooltypeid7', 'propertycountylandusecode',
    'propertylandusetypeid', 'propertyzoningdesc', 'regionidcountry',
    'regionidcity', 'regionidneighborhood', 'storytypeid',
    'storytypeid', 'typeconstructiontypeid', 'taxdelinquencyflag',
    'day', 'month',
]
# numerical features
NUMERICAL_FEATURES_NAMES = [
    'parcelid', 'logerror', 'transactiondate', 'basementsqft',
    'bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
    'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
    'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
    'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt',
    'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'latitude',
    'longitude', 'lotsizesquarefeet', 'poolcnt', 'poolsizesum',
    'rawcensustractandblock', 'regionidcounty', 'regionidzip', 'roomcnt',
    'threequarterbathnbr', 'unitcnt', 'yardbuildingsqft17',
    'yardbuildingsqft26', 'yearbuilt', 'numberofstories',
    'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear',
    'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyyear',
    'censustractandblock',
]

# target
TARGET_VAR = 'logerror'
