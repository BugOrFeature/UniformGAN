import pandas as pd
from rdt import HyperTransformer

from copulas.multivariate.gaussian import GaussianMultivariate

from model.ctabgan import CTABGAN


class UniformGAN(CTABGAN):
    def __init__(self, sdtypes, transformers, raw_csv_path, log_columns, mixed_columns, categorical_columns, problem_type):
        self.__name__ = 'CopulaCTABGAN'
        self.raw_cv_path = raw_csv_path
        self.raw_df = pd.read_csv(raw_csv_path)
        self.transformers = transformers
        self.log_columns = log_columns
        self.categorical_columns = categorical_columns
        self.mixed_columns = mixed_columns
        self.ht = HyperTransformer()
        self.ht.detect_initial_config(self.raw_df)
        self.ht.update_sdtypes(sdtypes)
        self.ht.update_transformers(transformers)
        self.ht.fit(self.raw_df)
        self.encoded_data = self.ht.transform(self.raw_df)
        try:
            self.categorical_columns = [v + '.value' for v in self.categorical_columns]
            self.log_columns = [v + '.value' for v in self.log_columns]
            self.mixed_columns = {k + '.value': [0.0] for k in self.mixed_columns}
        except KeyError as e:
            print("missing column; Error: " + str(e))

        super().__init__(
            mixed_columns=self.mixed_columns,
            categorical_columns=self.categorical_columns,
            log_columns=self.log_columns,
            raw_df=self.encoded_data,
            problem_type={list(problem_type.keys())[0]: problem_type[list(problem_type.keys())[0]] + '.value'},
        )

    def fit(self):
        super().fit()

    def get_config(self):
        return self.ht.get_config()

    def generate_samples(self):
        sample = super().generate_samples()
        sample = sample.astype(self.encoded_data.dtypes.to_dict())
        decoded_sample = self.ht.reverse_transform(sample)
        return decoded_sample
