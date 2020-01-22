def get_dataset(dataset_name):
    if dataset_name == 'cornell':
        from .cornell_data import CornellDataset
        return CornellDataset
    elif dataset_name == 'jacquard':
        from .jacquard_data import JacquardDataset
        return JacquardDataset
    elif dataset_name == 'rs_dir':
    	from .rs_dir_data import RsDirDataset
    	return RsDirDataset
    elif dataset_name == 'rs':
        from .rs_data import RsDataset
        return RsDataset
    else:
        raise NotImplementedError('Dataset Type {} is Not implemented'.format(dataset_name))