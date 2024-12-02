from .Loader import Loader
from torch.utils.data import ConcatDataset, DataLoader
from .batch_scheduler import BatchSchedulerSampler_ever2, SeqBatchSchedulerSampler, WeightBatchSchedulerSampler
from .data_loader import Dataloader
from .uea import collate_fn

train_data_dict_UEA = [
    'ArticularyWordRecognition',
    'AtrialFibrillation',
    'BasicMotions',
    'CharacterTrajectories',
    'Cricket' ,
    # 'DuckDuckGeese',
    # 'EigenWorms',
    'Epilepsy',
    'EthanolConcentration',
    'ERing',
    'FaceDetection',
    'FingerMovements',
    'HandMovementDirection',
    'Handwriting',
    'Heartbeat',
    'InsectWingbeat',
    'JapaneseVowels',
    'Libras',
    'LSST',
    'MotorImagery',
    'NATOPS',
    'PenDigits',
    'PEMS-SF',
    'PhonemeSpectra',
    'RacketSports',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'SpokenArabicDigits', 
    'StandWalkJump',
    'UWaveGestureLibrary',
]

train_data_dict_UCR = [
    'ACSF1',
    'Adiac',
    'AllGestureWiimoteX',
    'AllGestureWiimoteY',
    'AllGestureWiimoteZ',
    'ArrowHead',
    'Beef',
    'BeetleFly',
    'BirdChicken',
    'BME',
    'Car',
    'CBF',
    'Chinatown',
    'ChlorineConcentration',
    'CinCECGTorso',
    'Coffee',
    'Computers',
    'CricketX',
    'CricketY',
    'CricketZ',
    'Crop',
    'DiatomSizeReduction',
    'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW',
    'Earthquakes',
    'ECG200',
    'ECG5000',
    'ECGFiveDays',
    'ElectricDevices',
    'EOGHorizontalSignal',
    'EOGVerticalSignal',
    'EthanolLevel',
    'FaceAll',
    'FaceFour',
    'FacesUCR',
    'FiftyWords',
    'Fish',
    'FordA',
    'FordB',
    'FreezerRegularTrain',
    'FreezerSmallTrain',
    'Fungi',
    'GestureMidAirD1',
    'GestureMidAirD2',
    'GestureMidAirD3',
    'GesturePebbleZ1',
    'GesturePebbleZ2',
    'GunPoint',
    'GunPointAgeSpan',
    'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung',
    'Ham',
    'HandOutlines',
    'Haptics',
    'Herring',
    'HouseTwenty',
    'InlineSkate',
    'InsectEPGRegularTrain',
    'InsectEPGSmallTrain',
    'InsectWingbeatSound',
    'ItalyPowerDemand',
    'LargeKitchenAppliances',
    'Lightning2',
    'Lightning7',
    'Mallat',
    'Meat',
    'MedicalImages',
    'MelbournePedestrian',
    'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'MoteStrain',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'OliveOil',
    'OSULeaf',
    'PhalangesOutlinesCorrect',
    'Phoneme',
    'PickupGestureWiimoteZ',
    'PigAirwayPressure',
    'PigArtPressure',
    'PigCVP',
    'PLAID',
    'Plane',
    'PowerCons',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect',
    'ProximalPhalanxTW',
    'RefrigerationDevices',
    'Rock',
    'ScreenType',
    'SemgHandGenderCh2',
    'SemgHandMovementCh2',
    'SemgHandSubjectCh2',
    'ShakeGestureWiimoteZ',
    'ShapeletSim',
    'ShapesAll',
    'SmallKitchenAppliances',
    'SmoothSubspace',
    'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2',
    'StarLightCurves',
    'Strawberry',
    'SwedishLeaf',
    'Symbols',
    'SyntheticControl',
    'ToeSegmentation1',
    'ToeSegmentation2',
    'Trace',
    'TwoLeadECG',
    'TwoPatterns',
    'UMD',
    'UWaveGestureLibraryAll',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ',
    'Wafer',
    'Wine',
    'WordSynonyms',
    'Worms',
    'WormsTwoClass',
    'Yoga',
    'DodgerLoopDay',
    'DodgerLoopGame',
    'DodgerLoopWeekend',
]

train_data_dict_monash = [
    'AppliancesEnergy',
    'AustraliaRainfall',
    'BeijingPM10Quality',
    'BeijingPM25Quality',
    'BenzeneConcentration',
    'BIDMC32HR',
    'BIDMC32RR',
    'BIDMC32SpO2',
    'Covid3Month',
    'FloodModeling1',
    'FloodModeling2',
    'FloodModeling3',
    'HouseholdPowerConsumption1',
    'HouseholdPowerConsumption2',
    'IEEEPPG',
    'LiveFuelMoistureContent',
    'NewsHeadlineSentiment',
    'NewsTitleSentiment',
    'PPGDalia',
]

def train_data_provider(args, flag):
  
    if args.loader == 'UEA':
        train_data_dict = train_data_dict_UEA
    elif args.loader == 'UCR':
        train_data_dict = train_data_dict_UCR
    else:
        train_data_dict = train_data_dict_monash

    data_sum = len(train_data_dict)
    concat_dataset = []
    weights = []
    train_size = 0
    # config
    batch_size = 16
    drop_last = False

    i = 0
    for dataset_name in train_data_dict:
        i += 1
        dataset = Dataloader(
            loader = args.loader,
            dataset_name = dataset_name,
            flag=flag,
        )
        concat_dataset.append(dataset)
        weights.append(len(dataset))
        print(f'{dataset_name}  size: ', len(dataset))
        train_size += len(dataset)
        if i == data_sum:    break
    print('Pretrain data number: {0} | Pretrain data size: {1}'.format(data_sum, train_size))
    concat_dataset = ConcatDataset(concat_dataset)
    weights = [i / train_size for i in weights]

    data_loader = DataLoader(
        dataset=concat_dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=8,
        drop_last=drop_last,
        collate_fn=lambda x : collate_fn(x),
        # sampler=BatchSchedulerSampler_ever2(dataset=concat_dataset, batch_size=batch_size)
        # sampler=SeqBatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size)
        sampler=WeightBatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size, train_size=train_size, weights=weights)
    )
    return concat_dataset, data_loader


