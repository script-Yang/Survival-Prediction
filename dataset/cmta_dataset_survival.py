from __future__ import print_function, division
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv', modal='omic', apply_sig=False,
                 shuffle=False, seed=7, n_bins=4, ignore=[],
                 patient_strat=False, label_col=None, filter_dict={}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        print(csv_path)
        #slide_data = pd.read_csv(csv_path, low_memory=False)
        slide_data = pd.read_hdf(csv_path,'df')
        # slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        import pdb
        # pdb.set_trace()

        if not label_col:
            #label_col = 'survival_months'
            label_col = 'survival_days'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col


        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps

        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True,
                                     labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient: slide_ids})

        self.patient_dict = patient_dict

        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                label_dict.update({(i, c): key_count})
                key_count += 1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes = len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id': patients_df['case_id'].values, 'label': patients_df['label'].values}

        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:7]
        self.modal = modal
        self.cls_ids_prep()
        self.signatures = None
        # Signatures
        # self.apply_sig = apply_sig
        # if self.apply_sig:
        #     self.signatures = pd.read_csv('./csv/signatures.csv')
        # else:
        #     self.signatures = None

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id']))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]]  # get patient label
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def get_split_from_df(self, all_splits: dict, split_key: str = 'train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, modal=self.modal, signatures=self.signatures,
                                  data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None

        return split

    def return_splits(self, from_id: bool = True, csv_path: str = None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')

            # --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
        return train_split, val_split

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, modal = 'omic', OOM = 0, **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.modal = modal
        self.use_h5 = False
        self.OOM = OOM
        if self.OOM > 0:
            print('Using ramdomly sampled patches [{}] to avoid OOM error'.format(self.OOM))

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir

        if not self.use_h5:
            if self.data_dir:
                if self.modal == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        try:
                            wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                            wsi_bag = torch.load(wsi_path)
                            path_features.append(wsi_bag)
                        except FileNotFoundError:
                            continue
                    path_features = torch.cat(path_features, dim=0)
                    if self.OOM > 0:
                        if path_features.size(0) > self.OOM:
                            path_features = path_features[np.random.choice(path_features.size(0), self.OOM, replace=False)]
                    return (path_features, torch.zeros((1, 1)), label, event_time, c)

                elif self.modal == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        try:
                            wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                            wsi_bag = torch.load(wsi_path)
                            path_features.append(wsi_bag)
                            cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt'])
                        except FileNotFoundError:
                            print('FileNotFound: ', wsi_path)
                            continue
                    path_features = torch.cat(path_features, dim=0)
                    if self.OOM > 0:
                        if path_features.size(0) > self.OOM:
                            path_features = path_features[np.random.choice(path_features.size(0), self.OOM, replace=False)]
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.modal == 'omic':
                    # genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1, 1)), genomic_features, label, event_time, c)

                elif self.modal == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        try:
                            wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                            wsi_bag = torch.load(wsi_path)
                            path_features.append(wsi_bag)
                        except FileNotFoundError:
                            print('FileNotFound: ', wsi_path)
                            continue
                    path_features = torch.cat(path_features, dim=0)
                    if self.OOM > 0:
                        if path_features.size(0) > self.OOM:
                            path_features = path_features[np.random.choice(path_features.size(0), self.OOM, replace=False)]
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features, label, event_time, c)

                elif self.modal == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        try:
                            # wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                            # wsi_path = os.path.join(data_dir, '{}'.format(slide_id.rstrip('.svs')))
                            wsi_path = os.path.join(data_dir,slide_id)
                            wsi_bag = torch.load(wsi_path)
                            path_features.append(wsi_bag)
                        except FileNotFoundError:
                            print('FileNotFound: ', wsi_path)
                            continue
                    path_features = torch.cat(path_features, dim=0)
                    # if self.OOM > 0:
                    #     if path_features.size(0) > self.OOM:
                    #         path_features = path_features[np.random.choice(path_features.size(0), self.OOM, replace=False)]
                    #g_f1 = torch.tensor(self.genomic_features)
                    g_f = torch.tensor(self.genomic_features.iloc[idx].values)
                    # omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                    # omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    # omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    # omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    # omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    # omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    omic1 = omic2 = omic3 = omic4 = omic5 = omic6 = g_f
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)

                else:
                    raise NotImplementedError('Modal [%s] not implemented.' % self.modal)
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, modal, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.modal = modal
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]

        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        # --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic+modal for modal in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("Shape", self.genomic_features.shape)

    def __len__(self):
        return len(self.slide_data)

    # --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    # <--

    # --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple = None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    # <--
