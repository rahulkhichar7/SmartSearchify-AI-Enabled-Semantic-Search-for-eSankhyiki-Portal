PRODUCTS = ['ASI', 'ASUSE', 'CAMS', 'CPI', 'HCES', 'IIP', 'MIS', 'NAS', 'PLFS', 'WMI']
N_RESULTS = [3, 2, 1]  # Results per product tier
TOP_K_OVERALL = 10
TOP_K_CANDIDATES = 50
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
DATA_FILE = 'final_dataset.csv'
MODEL_FILE = 'LinearSVC_classifier.joblib'
