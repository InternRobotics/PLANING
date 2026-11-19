from dataloaders.DatasetSelfCaptured import SelfCapturedDataset


def load_dataset(args):
    if args.dataset_name != "selfCaptured":
        raise ValueError("This cleanup retains only the selfCaptured dataset.")
    return SelfCapturedDataset(args), False
