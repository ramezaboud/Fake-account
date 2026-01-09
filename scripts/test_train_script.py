import sys
import subprocess
import shutil
from pathlib import Path


def test_train_script_creates_pipeline(tmp_path):
    """Run a lightweight isolated copy of the project and check the pipeline is saved.

    This test copies the `src/` package into a temporary directory, writes tiny
    `data/labeled_dataset.csv`, provides a stubbed
    `sklearn.model_selection.GridSearchCV` implementation to avoid long
    hyperparameter searches, runs `python src/train.py`, and asserts that
    `models/randomforest_pipeline.joblib` is created.
    """
    # repo root is one level above tests/ in this workspace
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / 'src'

    # create temp project layout
    project_tmp = tmp_path / 'project'
    project_tmp.mkdir()

    # copy src folder
    shutil.copytree(src_dir, project_tmp / 'src')

    # create small data folder with minimal rows (labeled_dataset.csv)
    data_dir = project_tmp / 'data'
    data_dir.mkdir()
    labeled_csv = data_dir / 'labeled_dataset.csv'

    header = (
        'lang,created_at,statuses_count,followers_count,friends_count,'
        'favourites_count,listed_count,name,description,default_profile,verified,label\n'
    )
    # four small rows with labels (0 = real, 1 = fake)
    labeled_csv.write_text(
        header +
        'en,2020-01-01 00:00:00,100,50,20,5,1,John Doe,hello,0,0,0\n'
        'fr,2019-06-15 12:34:56,200,150,40,10,2,Jane Roe,hi,0,1,0\n'
        'es,2018-03-03 03:03:03,10,5,2,0,0,Faux One,fake,1,0,1\n'
        'pt,2017-07-07 07:07:07,5,2,1,0,0,Faux Two,bot,1,0,1\n'
    )

    # create a lightweight stub module and patch train.py to import it
    # instead of the real sklearn.model_selection to avoid circular import
    stub_code = '''
def train_test_split(X, y, test_size=0.25, random_state=None, stratify=None):
    # simple deterministic split: first N -> train, rest -> test
    n = len(X)
    test_n = int(n * test_size)
    # keep order for determinism in this test
    X_train = X.iloc[test_n:]
    X_test = X.iloc[:test_n]
    y_train = y[test_n:]
    y_test = y[:test_n]
    return X_train, X_test, y_train, y_test

class GridSearchCV:
    def __init__(self, estimator, param_grid=None, cv=2,
                 n_jobs=None, scoring=None, verbose=0):
        self.estimator = estimator
        self.best_estimator_ = None
        self.best_params_ = {}
        self.best_score_ = 0.0

    def fit(self, X, y):
        # directly fit the provided estimator and return it as best
        self.estimator.fit(X, y)
        self.best_estimator_ = self.estimator
        self.best_score_ = 0.85  # mock score
        return self
'''
    stub_path = project_tmp / 'sklearn_model_selection_stub.py'
    stub_path.write_text(stub_code)

    # patch train.py import to use the stub module
    train_file = project_tmp / 'src' / 'train.py'
    train_text = train_file.read_text()
    train_text = train_text.replace(
        'from sklearn.model_selection import train_test_split, GridSearchCV',
        'from sklearn_model_selection_stub import train_test_split, GridSearchCV'
    )
    train_file.write_text(train_text)

    # patch roc_auc_score call to be safe when only one class present in y_test
    train_text = train_file.read_text()
    train_text = train_text.replace(
        "if y_proba is not None:\n    print(f'Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}')",
        "if y_proba is not None:\n    try:\n        print(f'Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}')\n    except ValueError:\n        print('Test ROC AUC: undefined (only one class in y_test)')"
    )
    train_file.write_text(train_text)

    # run the training script inside the temp project
    env = dict(**{k: v for k, v in {}.items()})
    # ensure Python uses the temp project first on PYTHONPATH
    env_py = str(project_tmp)
    # preserve current env
    env = dict(**{**shutil.os.environ})
    env['PYTHONPATH'] = env_py + (shutil.os.pathsep + env.get('PYTHONPATH', ''))

    # run the script
    completed = subprocess.run([sys.executable, 'src/train.py'], cwd=str(project_tmp), env=env, capture_output=True, text=True)

    # for debugging, if failed print stdout/stderr
    if completed.returncode != 0:
        print('STDOUT:\n', completed.stdout)
        print('STDERR:\n', completed.stderr)

    assert completed.returncode == 0, f"train.py failed (see stdout/stderr)"

    # assert the model file exists
    model_file = project_tmp / 'models' / 'randomforest_pipeline.joblib'
    assert model_file.exists(), f"Expected model file at {model_file}"
