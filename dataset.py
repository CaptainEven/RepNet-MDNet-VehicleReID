# coding=utf-8

import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys
import pickle
import random
import shutil
from skimage.transform import resize
from tqdm import tqdm

id2name = {0: 'tench, Tinca tinca',
           1: 'goldfish, Carassius auratus',
           2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
           3: 'tiger shark, Galeocerdo cuvieri',
           4: 'hammerhead, hammerhead shark',
           5: 'electric ray, crampfish, numbfish, torpedo',
           6: 'stingray',
           7: 'cock',
           8: 'hen',
           9: 'ostrich, Struthio camelus',
           10: 'brambling, Fringilla montifringilla',
           11: 'goldfinch, Carduelis carduelis',
           12: 'house finch, linnet, Carpodacus mexicanus',
           13: 'junco, snowbird',
           14: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
           15: 'robin, American robin, Turdus migratorius',
           16: 'bulbul',
           17: 'jay',
           18: 'magpie',
           19: 'chickadee',
           20: 'water ouzel, dipper',
           21: 'kite',
           22: 'bald eagle, American eagle, Haliaeetus leucocephalus',
           23: 'vulture',
           24: 'great grey owl, great gray owl, Strix nebulosa',
           25: 'European fire salamander, Salamandra salamandra',
           26: 'common newt, Triturus vulgaris',
           27: 'eft',
           28: 'spotted salamander, Ambystoma maculatum',
           29: 'axolotl, mud puppy, Ambystoma mexicanum',
           30: 'bullfrog, Rana catesbeiana',
           31: 'tree frog, tree-frog',
           32: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
           33: 'loggerhead, loggerhead turtle, Caretta caretta',
           34: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
           35: 'mud turtle',
           36: 'terrapin',
           37: 'box turtle, box tortoise',
           38: 'banded gecko',
           39: 'common iguana, iguana, Iguana iguana',
           40: 'American chameleon, anole, Anolis carolinensis',
           41: 'whiptail, whiptail lizard',
           42: 'agama',
           43: 'frilled lizard, Chlamydosaurus kingi',
           44: 'alligator lizard',
           45: 'Gila monster, Heloderma suspectum',
           46: 'green lizard, Lacerta viridis',
           47: 'African chameleon, Chamaeleo chamaeleon',
           48: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
           49: 'African crocodile, Nile crocodile, Crocodylus niloticus',
           50: 'American alligator, Alligator mississipiensis',
           51: 'triceratops',
           52: 'thunder snake, worm snake, Carphophis amoenus',
           53: 'ringneck snake, ring-necked snake, ring snake',
           54: 'hognose snake, puff adder, sand viper',
           55: 'green snake, grass snake',
           56: 'king snake, kingsnake',
           57: 'garter snake, grass snake',
           58: 'water snake',
           59: 'vine snake',
           60: 'night snake, Hypsiglena torquata',
           61: 'boa constrictor, Constrictor constrictor',
           62: 'rock python, rock snake, Python sebae',
           63: 'Indian cobra, Naja naja',
           64: 'green mamba',
           65: 'sea snake',
           66: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
           67: 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
           68: 'sidewinder, horned rattlesnake, Crotalus cerastes',
           69: 'trilobite',
           70: 'harvestman, daddy longlegs, Phalangium opilio',
           71: 'scorpion',
           72: 'black and gold garden spider, Argiope aurantia',
           73: 'barn spider, Araneus cavaticus',
           74: 'garden spider, Aranea diademata',
           75: 'black widow, Latrodectus mactans',
           76: 'tarantula',
           77: 'wolf spider, hunting spider',
           78: 'tick',
           79: 'centipede',
           80: 'black grouse',
           81: 'ptarmigan',
           82: 'ruffed grouse, partridge, Bonasa umbellus',
           83: 'prairie chicken, prairie grouse, prairie fowl',
           84: 'peacock',
           85: 'quail',
           86: 'partridge',
           87: 'African grey, African gray, Psittacus erithacus',
           88: 'macaw',
           89: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
           90: 'lorikeet',
           91: 'coucal',
           92: 'bee eater',
           93: 'hornbill',
           94: 'hummingbird',
           95: 'jacamar',
           96: 'toucan',
           97: 'drake',
           98: 'red-breasted merganser, Mergus serrator',
           99: 'goose',
           100: 'black swan, Cygnus atratus',
           101: 'tusker',
           102: 'echidna, spiny anteater, anteater',
           103: 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
           104: 'wallaby, brush kangaroo',
           105: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
           106: 'wombat',
           107: 'jellyfish',
           108: 'sea anemone, anemone',
           109: 'brain coral',
           110: 'flatworm, platyhelminth',
           111: 'nematode, nematode worm, roundworm',
           112: 'conch',
           113: 'snail',
           114: 'slug',
           115: 'sea slug, nudibranch',
           116: 'chiton, coat-of-mail shell, sea cradle, polyplacophore',
           117: 'chambered nautilus, pearly nautilus, nautilus',
           118: 'Dungeness crab, Cancer magister',
           119: 'rock crab, Cancer irroratus',
           120: 'fiddler crab',
           121: 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
           122: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
           123: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
           124: 'crayfish, crawfish, crawdad, crawdaddy',
           125: 'hermit crab',
           126: 'isopod',
           127: 'white stork, Ciconia ciconia',
           128: 'black stork, Ciconia nigra',
           129: 'spoonbill',
           130: 'flamingo',
           131: 'little blue heron, Egretta caerulea',
           132: 'American egret, great white heron, Egretta albus',
           133: 'bittern',
           134: 'crane',
           135: 'limpkin, Aramus pictus',
           136: 'European gallinule, Porphyrio porphyrio',
           137: 'American coot, marsh hen, mud hen, water hen, Fulica americana',
           138: 'bustard',
           139: 'ruddy turnstone, Arenaria interpres',
           140: 'red-backed sandpiper, dunlin, Erolia alpina',
           141: 'redshank, Tringa totanus',
           142: 'dowitcher',
           143: 'oystercatcher, oyster catcher',
           144: 'pelican',
           145: 'king penguin, Aptenodytes patagonica',
           146: 'albatross, mollymawk',
           147: 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
           148: 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
           149: 'dugong, Dugong dugon',
           150: 'sea lion',
           151: 'Chihuahua',
           152: 'Japanese spaniel',
           153: 'Maltese dog, Maltese terrier, Maltese',
           154: 'Pekinese, Pekingese, Peke',
           155: 'Shih-Tzu',
           156: 'Blenheim spaniel',
           157: 'papillon',
           158: 'toy terrier',
           159: 'Rhodesian ridgeback',
           160: 'Afghan hound, Afghan',
           161: 'basset, basset hound',
           162: 'beagle',
           163: 'bloodhound, sleuthhound',
           164: 'bluetick',
           165: 'black-and-tan coonhound',
           166: 'Walker hound, Walker foxhound',
           167: 'English foxhound',
           168: 'redbone',
           169: 'borzoi, Russian wolfhound',
           170: 'Irish wolfhound',
           171: 'Italian greyhound',
           172: 'whippet',
           173: 'Ibizan hound, Ibizan Podenco',
           174: 'Norwegian elkhound, elkhound',
           175: 'otterhound, otter hound',
           176: 'Saluki, gazelle hound',
           177: 'Scottish deerhound, deerhound',
           178: 'Weimaraner',
           179: 'Staffordshire bullterrier, Staffordshire bull terrier',
           180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
           181: 'Bedlington terrier',
           182: 'Border terrier',
           183: 'Kerry blue terrier',
           184: 'Irish terrier',
           185: 'Norfolk terrier',
           186: 'Norwich terrier',
           187: 'Yorkshire terrier',
           188: 'wire-haired fox terrier',
           189: 'Lakeland terrier',
           190: 'Sealyham terrier, Sealyham',
           191: 'Airedale, Airedale terrier',
           192: 'cairn, cairn terrier',
           193: 'Australian terrier',
           194: 'Dandie Dinmont, Dandie Dinmont terrier',
           195: 'Boston bull, Boston terrier',
           196: 'miniature schnauzer',
           197: 'giant schnauzer',
           198: 'standard schnauzer',
           199: 'Scotch terrier, Scottish terrier, Scottie',
           200: 'Tibetan terrier, chrysanthemum dog',
           201: 'silky terrier, Sydney silky',
           202: 'soft-coated wheaten terrier',
           203: 'West Highland white terrier',
           204: 'Lhasa, Lhasa apso',
           205: 'flat-coated retriever',
           206: 'curly-coated retriever',
           207: 'golden retriever',
           208: 'Labrador retriever',
           209: 'Chesapeake Bay retriever',
           210: 'German short-haired pointer',
           211: 'vizsla, Hungarian pointer',
           212: 'English setter',
           213: 'Irish setter, red setter',
           214: 'Gordon setter',
           215: 'Brittany spaniel',
           216: 'clumber, clumber spaniel',
           217: 'English springer, English springer spaniel',
           218: 'Welsh springer spaniel',
           219: 'cocker spaniel, English cocker spaniel, cocker',
           220: 'Sussex spaniel',
           221: 'Irish water spaniel',
           222: 'kuvasz',
           223: 'schipperke',
           224: 'groenendael',
           225: 'malinois',
           226: 'briard',
           227: 'kelpie',
           228: 'komondor',
           229: 'Old English sheepdog, bobtail',
           230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
           231: 'collie',
           232: 'Border collie',
           233: 'Bouvier des Flandres, Bouviers des Flandres',
           234: 'Rottweiler',
           235: 'German shepherd, German shepherd dog, German police dog, alsatian',
           236: 'Doberman, Doberman pinscher',
           237: 'miniature pinscher',
           238: 'Greater Swiss Mountain dog',
           239: 'Bernese mountain dog',
           240: 'Appenzeller',
           241: 'EntleBucher',
           242: 'boxer',
           243: 'bull mastiff',
           244: 'Tibetan mastiff',
           245: 'French bulldog',
           246: 'Great Dane',
           247: 'Saint Bernard, St Bernard',
           248: 'Eskimo dog, husky',
           249: 'malamute, malemute, Alaskan malamute',
           250: 'Siberian husky',
           251: 'dalmatian, coach dog, carriage dog',
           252: 'affenpinscher, monkey pinscher, monkey dog',
           253: 'basenji',
           254: 'pug, pug-dog',
           255: 'Leonberg',
           256: 'Newfoundland, Newfoundland dog',
           257: 'Great Pyrenees',
           258: 'Samoyed, Samoyede',
           259: 'Pomeranian',
           260: 'chow, chow chow',
           261: 'keeshond',
           262: 'Brabancon griffon',
           263: 'Pembroke, Pembroke Welsh corgi',
           264: 'Cardigan, Cardigan Welsh corgi',
           265: 'toy poodle',
           266: 'miniature poodle',
           267: 'standard poodle',
           268: 'Mexican hairless',
           269: 'timber wolf, grey wolf, gray wolf, Canis lupus',
           270: 'white wolf, Arctic wolf, Canis lupus tundrarum',
           271: 'red wolf, maned wolf, Canis rufus, Canis niger',
           272: 'coyote, prairie wolf, brush wolf, Canis latrans',
           273: 'dingo, warrigal, warragal, Canis dingo',
           274: 'dhole, Cuon alpinus',
           275: 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
           276: 'hyena, hyaena',
           277: 'red fox, Vulpes vulpes',
           278: 'kit fox, Vulpes macrotis',
           279: 'Arctic fox, white fox, Alopex lagopus',
           280: 'grey fox, gray fox, Urocyon cinereoargenteus',
           281: 'tabby, tabby cat',
           282: 'tiger cat',
           283: 'Persian cat',
           284: 'Siamese cat, Siamese',
           285: 'Egyptian cat',
           286: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
           287: 'lynx, catamount',
           288: 'leopard, Panthera pardus',
           289: 'snow leopard, ounce, Panthera uncia',
           290: 'jaguar, panther, Panthera onca, Felis onca',
           291: 'lion, king of beasts, Panthera leo',
           292: 'tiger, Panthera tigris',
           293: 'cheetah, chetah, Acinonyx jubatus',
           294: 'brown bear, bruin, Ursus arctos',
           295: 'American black bear, black bear, Ursus americanus, Euarctos americanus',
           296: 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
           297: 'sloth bear, Melursus ursinus, Ursus ursinus',
           298: 'mongoose',
           299: 'meerkat, mierkat',
           300: 'tiger beetle',
           301: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
           302: 'ground beetle, carabid beetle',
           303: 'long-horned beetle, longicorn, longicorn beetle',
           304: 'leaf beetle, chrysomelid',
           305: 'dung beetle',
           306: 'rhinoceros beetle',
           307: 'weevil',
           308: 'fly',
           309: 'bee',
           310: 'ant, emmet, pismire',
           311: 'grasshopper, hopper',
           312: 'cricket',
           313: 'walking stick, walkingstick, stick insect',
           314: 'cockroach, roach',
           315: 'mantis, mantid',
           316: 'cicada, cicala',
           317: 'leafhopper',
           318: 'lacewing, lacewing fly',
           319: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
           320: 'damselfly',
           321: 'admiral',
           322: 'ringlet, ringlet butterfly',
           323: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
           324: 'cabbage butterfly',
           325: 'sulphur butterfly, sulfur butterfly',
           326: 'lycaenid, lycaenid butterfly',
           327: 'starfish, sea star',
           328: 'sea urchin',
           329: 'sea cucumber, holothurian',
           330: 'wood rabbit, cottontail, cottontail rabbit',
           331: 'hare',
           332: 'Angora, Angora rabbit',
           333: 'hamster',
           334: 'porcupine, hedgehog',
           335: 'fox squirrel, eastern fox squirrel, Sciurus niger',
           336: 'marmot',
           337: 'beaver',
           338: 'guinea pig, Cavia cobaya',
           339: 'sorrel',
           340: 'zebra',
           341: 'hog, pig, grunter, squealer, Sus scrofa',
           342: 'wild boar, boar, Sus scrofa',
           343: 'warthog',
           344: 'hippopotamus, hippo, river horse, Hippopotamus amphibius',
           345: 'ox',
           346: 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
           347: 'bison',
           348: 'ram, tup',
           349: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
           350: 'ibex, Capra ibex',
           351: 'hartebeest',
           352: 'impala, Aepyceros melampus',
           353: 'gazelle',
           354: 'Arabian camel, dromedary, Camelus dromedarius',
           355: 'llama',
           356: 'weasel',
           357: 'mink',
           358: 'polecat, fitch, foulmart, foumart, Mustela putorius',
           359: 'black-footed ferret, ferret, Mustela nigripes',
           360: 'otter',
           361: 'skunk, polecat, wood pussy',
           362: 'badger',
           363: 'armadillo',
           364: 'three-toed sloth, ai, Bradypus tridactylus',
           365: 'orangutan, orang, orangutang, Pongo pygmaeus',
           366: 'gorilla, Gorilla gorilla',
           367: 'chimpanzee, chimp, Pan troglodytes',
           368: 'gibbon, Hylobates lar',
           369: 'siamang, Hylobates syndactylus, Symphalangus syndactylus',
           370: 'guenon, guenon monkey',
           371: 'patas, hussar monkey, Erythrocebus patas',
           372: 'baboon',
           373: 'macaque',
           374: 'langur',
           375: 'colobus, colobus monkey',
           376: 'proboscis monkey, Nasalis larvatus',
           377: 'marmoset',
           378: 'capuchin, ringtail, Cebus capucinus',
           379: 'howler monkey, howler',
           380: 'titi, titi monkey',
           381: 'spider monkey, Ateles geoffroyi',
           382: 'squirrel monkey, Saimiri sciureus',
           383: 'Madagascar cat, ring-tailed lemur, Lemur catta',
           384: 'indri, indris, Indri indri, Indri brevicaudatus',
           385: 'Indian elephant, Elephas maximus',
           386: 'African elephant, Loxodonta africana',
           387: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
           388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
           389: 'barracouta, snoek',
           390: 'eel',
           391: 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
           392: 'rock beauty, Holocanthus tricolor',
           393: 'anemone fish',
           394: 'sturgeon',
           395: 'gar, garfish, garpike, billfish, Lepisosteus osseus',
           396: 'lionfish',
           397: 'puffer, pufferfish, blowfish, globefish',
           398: 'abacus',
           399: 'abaya',
           400: "academic gown, academic robe, judge's robe",
           401: 'accordion, piano accordion, squeeze box',
           402: 'acoustic guitar',
           403: 'aircraft carrier, carrier, flattop, attack aircraft carrier',
           404: 'airliner',
           405: 'airship, dirigible',
           406: 'altar',
           407: 'ambulance',
           408: 'amphibian, amphibious vehicle',
           409: 'analog clock',
           410: 'apiary, bee house',
           411: 'apron',
           412: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
           413: 'assault rifle, assault gun',
           414: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
           415: 'bakery, bakeshop, bakehouse',
           416: 'balance beam, beam',
           417: 'balloon',
           418: 'ballpoint, ballpoint pen, ballpen, Biro',
           419: 'Band Aid',
           420: 'banjo',
           421: 'bannister, banister, balustrade, balusters, handrail',
           422: 'barbell',
           423: 'barber chair',
           424: 'barbershop',
           425: 'barn',
           426: 'barometer',
           427: 'barrel, cask',
           428: 'barrow, garden cart, lawn cart, wheelbarrow',
           429: 'baseball',
           430: 'basketball',
           431: 'bassinet',
           432: 'bassoon',
           433: 'bathing cap, swimming cap',
           434: 'bath towel',
           435: 'bathtub, bathing tub, bath, tub',
           436: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
           437: 'beacon, lighthouse, beacon light, pharos',
           438: 'beaker',
           439: 'bearskin, busby, shako',
           440: 'beer bottle',
           441: 'beer glass',
           442: 'bell cote, bell cot',
           443: 'bib',
           444: 'bicycle-built-for-two, tandem bicycle, tandem',
           445: 'bikini, two-piece',
           446: 'binder, ring-binder',
           447: 'binoculars, field glasses, opera glasses',
           448: 'birdhouse',
           449: 'boathouse',
           450: 'bobsled, bobsleigh, bob',
           451: 'bolo tie, bolo, bola tie, bola',
           452: 'bonnet, poke bonnet',
           453: 'bookcase',
           454: 'bookshop, bookstore, bookstall',
           455: 'bottlecap',
           456: 'bow',
           457: 'bow tie, bow-tie, bowtie',
           458: 'brass, memorial tablet, plaque',
           459: 'brassiere, bra, bandeau',
           460: 'breakwater, groin, groyne, mole, bulwark, seawall, jetty',
           461: 'breastplate, aegis, egis',
           462: 'broom',
           463: 'bucket, pail',
           464: 'buckle',
           465: 'bulletproof vest',
           466: 'bullet train, bullet',
           467: 'butcher shop, meat market',
           468: 'cab, hack, taxi, taxicab',
           469: 'caldron, cauldron',
           470: 'candle, taper, wax light',
           471: 'cannon',
           472: 'canoe',
           473: 'can opener, tin opener',
           474: 'cardigan',
           475: 'car mirror',
           476: 'carousel, carrousel, merry-go-round, roundabout, whirligig',
           477: "carpenter's kit, tool kit",
           478: 'carton',
           479: 'car wheel',
           480: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
           481: 'cassette',
           482: 'cassette player',
           483: 'castle',
           484: 'catamaran',
           485: 'CD player',
           486: 'cello, violoncello',
           487: 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
           488: 'chain',
           489: 'chainlink fence',
           490: 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour',
           491: 'chain saw, chainsaw',
           492: 'chest',
           493: 'chiffonier, commode',
           494: 'chime, bell, gong',
           495: 'china cabinet, china closet',
           496: 'Christmas stocking',
           497: 'church, church building',
           498: 'cinema, movie theater, movie theatre, movie house, picture palace',
           499: 'cleaver, meat cleaver, chopper',
           500: 'cliff dwelling',
           501: 'cloak',
           502: 'clog, geta, patten, sabot',
           503: 'cocktail shaker',
           504: 'coffee mug',
           505: 'coffeepot',
           506: 'coil, spiral, volute, whorl, helix',
           507: 'combination lock',
           508: 'computer keyboard, keypad',
           509: 'confectionery, confectionary, candy store',
           510: 'container ship, containership, container vessel',
           511: 'convertible',
           512: 'corkscrew, bottle screw',
           513: 'cornet, horn, trumpet, trump',
           514: 'cowboy boot',
           515: 'cowboy hat, ten-gallon hat',
           516: 'cradle',
           517: 'crane',
           518: 'crash helmet',
           519: 'crate',
           520: 'crib, cot',
           521: 'Crock Pot',
           522: 'croquet ball',
           523: 'crutch',
           524: 'cuirass',
           525: 'dam, dike, dyke',
           526: 'desk',
           527: 'desktop computer',
           528: 'dial telephone, dial phone',
           529: 'diaper, nappy, napkin',
           530: 'digital clock',
           531: 'digital watch',
           532: 'dining table, board',
           533: 'dishrag, dishcloth',
           534: 'dishwasher, dish washer, dishwashing machine',
           535: 'disk brake, disc brake',
           536: 'dock, dockage, docking facility',
           537: 'dogsled, dog sled, dog sleigh',
           538: 'dome',
           539: 'doormat, welcome mat',
           540: 'drilling platform, offshore rig',
           541: 'drum, membranophone, tympan',
           542: 'drumstick',
           543: 'dumbbell',
           544: 'Dutch oven',
           545: 'electric fan, blower',
           546: 'electric guitar',
           547: 'electric locomotive',
           548: 'entertainment center',
           549: 'envelope',
           550: 'espresso maker',
           551: 'face powder',
           552: 'feather boa, boa',
           553: 'file, file cabinet, filing cabinet',
           554: 'fireboat',
           555: 'fire engine, fire truck',
           556: 'fire screen, fireguard',
           557: 'flagpole, flagstaff',
           558: 'flute, transverse flute',
           559: 'folding chair',
           560: 'football helmet',
           561: 'forklift',
           562: 'fountain',
           563: 'fountain pen',
           564: 'four-poster',
           565: 'freight car',
           566: 'French horn, horn',
           567: 'frying pan, frypan, skillet',
           568: 'fur coat',
           569: 'garbage truck, dustcart',
           570: 'gasmask, respirator, gas helmet',
           571: 'gas pump, gasoline pump, petrol pump, island dispenser',
           572: 'goblet',
           573: 'go-kart',
           574: 'golf ball',
           575: 'golfcart, golf cart',
           576: 'gondola',
           577: 'gong, tam-tam',
           578: 'gown',
           579: 'grand piano, grand',
           580: 'greenhouse, nursery, glasshouse',
           581: 'grille, radiator grille',
           582: 'grocery store, grocery, food market, market',
           583: 'guillotine',
           584: 'hair slide',
           585: 'hair spray',
           586: 'half track',
           587: 'hammer',
           588: 'hamper',
           589: 'hand blower, blow dryer, blow drier, hair dryer, hair drier',
           590: 'hand-held computer, hand-held microcomputer',
           591: 'handkerchief, hankie, hanky, hankey',
           592: 'hard disc, hard disk, fixed disk',
           593: 'harmonica, mouth organ, harp, mouth harp',
           594: 'harp',
           595: 'harvester, reaper',
           596: 'hatchet',
           597: 'holster',
           598: 'home theater, home theatre',
           599: 'honeycomb',
           600: 'hook, claw',
           601: 'hoopskirt, crinoline',
           602: 'horizontal bar, high bar',
           603: 'horse cart, horse-cart',
           604: 'hourglass',
           605: 'iPod',
           606: 'iron, smoothing iron',
           607: "jack-o'-lantern",
           608: 'jean, blue jean, denim',
           609: 'jeep, landrover',
           610: 'jersey, T-shirt, tee shirt',
           611: 'jigsaw puzzle',
           612: 'jinrikisha, ricksha, rickshaw',
           613: 'joystick',
           614: 'kimono',
           615: 'knee pad',
           616: 'knot',
           617: 'lab coat, laboratory coat',
           618: 'ladle',
           619: 'lampshade, lamp shade',
           620: 'laptop, laptop computer',
           621: 'lawn mower, mower',
           622: 'lens cap, lens cover',
           623: 'letter opener, paper knife, paperknife',
           624: 'library',
           625: 'lifeboat',
           626: 'lighter, light, igniter, ignitor',
           627: 'limousine, limo',
           628: 'liner, ocean liner',
           629: 'lipstick, lip rouge',
           630: 'Loafer',
           631: 'lotion',
           632: 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
           633: "loupe, jeweler's loupe",
           634: 'lumbermill, sawmill',
           635: 'magnetic compass',
           636: 'mailbag, postbag',
           637: 'mailbox, letter box',
           638: 'maillot',
           639: 'maillot, tank suit',
           640: 'manhole cover',
           641: 'maraca',
           642: 'marimba, xylophone',
           643: 'mask',
           644: 'matchstick',
           645: 'maypole',
           646: 'maze, labyrinth',
           647: 'measuring cup',
           648: 'medicine chest, medicine cabinet',
           649: 'megalith, megalithic structure',
           650: 'microphone, mike',
           651: 'microwave, microwave oven',
           652: 'military uniform',
           653: 'milk can',
           654: 'minibus',
           655: 'miniskirt, mini',
           656: 'minivan',
           657: 'missile',
           658: 'mitten',
           659: 'mixing bowl',
           660: 'mobile home, manufactured home',
           661: 'Model T',
           662: 'modem',
           663: 'monastery',
           664: 'monitor',
           665: 'moped',
           666: 'mortar',
           667: 'mortarboard',
           668: 'mosque',
           669: 'mosquito net',
           670: 'motor scooter, scooter',
           671: 'mountain bike, all-terrain bike, off-roader',
           672: 'mountain tent',
           673: 'mouse, computer mouse',
           674: 'mousetrap',
           675: 'moving van',
           676: 'muzzle',
           677: 'nail',
           678: 'neck brace',
           679: 'necklace',
           680: 'nipple',
           681: 'notebook, notebook computer',
           682: 'obelisk',
           683: 'oboe, hautboy, hautbois',
           684: 'ocarina, sweet potato',
           685: 'odometer, hodometer, mileometer, milometer',
           686: 'oil filter',
           687: 'organ, pipe organ',
           688: 'oscilloscope, scope, cathode-ray oscilloscope, CRO',
           689: 'overskirt',
           690: 'oxcart',
           691: 'oxygen mask',
           692: 'packet',
           693: 'paddle, boat paddle',
           694: 'paddlewheel, paddle wheel',
           695: 'padlock',
           696: 'paintbrush',
           697: "pajama, pyjama, pj's, jammies",
           698: 'palace',
           699: 'panpipe, pandean pipe, syrinx',
           700: 'paper towel',
           701: 'parachute, chute',
           702: 'parallel bars, bars',
           703: 'park bench',
           704: 'parking meter',
           705: 'passenger car, coach, carriage',
           706: 'patio, terrace',
           707: 'pay-phone, pay-station',
           708: 'pedestal, plinth, footstall',
           709: 'pencil box, pencil case',
           710: 'pencil sharpener',
           711: 'perfume, essence',
           712: 'Petri dish',
           713: 'photocopier',
           714: 'pick, plectrum, plectron',
           715: 'pickelhaube',
           716: 'picket fence, paling',
           717: 'pickup, pickup truck',
           718: 'pier',
           719: 'piggy bank, penny bank',
           720: 'pill bottle',
           721: 'pillow',
           722: 'ping-pong ball',
           723: 'pinwheel',
           724: 'pirate, pirate ship',
           725: 'pitcher, ewer',
           726: "plane, carpenter's plane, woodworking plane",
           727: 'planetarium',
           728: 'plastic bag',
           729: 'plate rack',
           730: 'plow, plough',
           731: "plunger, plumber's helper",
           732: 'Polaroid camera, Polaroid Land camera',
           733: 'pole',
           734: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
           735: 'poncho',
           736: 'pool table, billiard table, snooker table',
           737: 'pop bottle, soda bottle',
           738: 'pot, flowerpot',
           739: "potter's wheel",
           740: 'power drill',
           741: 'prayer rug, prayer mat',
           742: 'printer',
           743: 'prison, prison house',
           744: 'projectile, missile',
           745: 'projector',
           746: 'puck, hockey puck',
           747: 'punching bag, punch bag, punching ball, punchball',
           748: 'purse',
           749: 'quill, quill pen',
           750: 'quilt, comforter, comfort, puff',
           751: 'racer, race car, racing car',
           752: 'racket, racquet',
           753: 'radiator',
           754: 'radio, wireless',
           755: 'radio telescope, radio reflector',
           756: 'rain barrel',
           757: 'recreational vehicle, RV, R.V.',
           758: 'reel',
           759: 'reflex camera',
           760: 'refrigerator, icebox',
           761: 'remote control, remote',
           762: 'restaurant, eating house, eating place, eatery',
           763: 'revolver, six-gun, six-shooter',
           764: 'rifle',
           765: 'rocking chair, rocker',
           766: 'rotisserie',
           767: 'rubber eraser, rubber, pencil eraser',
           768: 'rugby ball',
           769: 'rule, ruler',
           770: 'running shoe',
           771: 'safe',
           772: 'safety pin',
           773: 'saltshaker, salt shaker',
           774: 'sandal',
           775: 'sarong',
           776: 'sax, saxophone',
           777: 'scabbard',
           778: 'scale, weighing machine',
           779: 'school bus',
           780: 'schooner',
           781: 'scoreboard',
           782: 'screen, CRT screen',
           783: 'screw',
           784: 'screwdriver',
           785: 'seat belt, seatbelt',
           786: 'sewing machine',
           787: 'shield, buckler',
           788: 'shoe shop, shoe-shop, shoe store',
           789: 'shoji',
           790: 'shopping basket',
           791: 'shopping cart',
           792: 'shovel',
           793: 'shower cap',
           794: 'shower curtain',
           795: 'ski',
           796: 'ski mask',
           797: 'sleeping bag',
           798: 'slide rule, slipstick',
           799: 'sliding door',
           800: 'slot, one-armed bandit',
           801: 'snorkel',
           802: 'snowmobile',
           803: 'snowplow, snowplough',
           804: 'soap dispenser',
           805: 'soccer ball',
           806: 'sock',
           807: 'solar dish, solar collector, solar furnace',
           808: 'sombrero',
           809: 'soup bowl',
           810: 'space bar',
           811: 'space heater',
           812: 'space shuttle',
           813: 'spatula',
           814: 'speedboat',
           815: "spider web, spider's web",
           816: 'spindle',
           817: 'sports car, sport car',
           818: 'spotlight, spot',
           819: 'stage',
           820: 'steam locomotive',
           821: 'steel arch bridge',
           822: 'steel drum',
           823: 'stethoscope',
           824: 'stole',
           825: 'stone wall',
           826: 'stopwatch, stop watch',
           827: 'stove',
           828: 'strainer',
           829: 'streetcar, tram, tramcar, trolley, trolley car',
           830: 'stretcher',
           831: 'studio couch, day bed',
           832: 'stupa, tope',
           833: 'submarine, pigboat, sub, U-boat',
           834: 'suit, suit of clothes',
           835: 'sundial',
           836: 'sunglass',
           837: 'sunglasses, dark glasses, shades',
           838: 'sunscreen, sunblock, sun blocker',
           839: 'suspension bridge',
           840: 'swab, swob, mop',
           841: 'sweatshirt',
           842: 'swimming trunks, bathing trunks',
           843: 'swing',
           844: 'switch, electric switch, electrical switch',
           845: 'syringe',
           846: 'table lamp',
           847: 'tank, army tank, armored combat vehicle, armoured combat vehicle',
           848: 'tape player',
           849: 'teapot',
           850: 'teddy, teddy bear',
           851: 'television, television system',
           852: 'tennis ball',
           853: 'thatch, thatched roof',
           854: 'theater curtain, theatre curtain',
           855: 'thimble',
           856: 'thresher, thrasher, threshing machine',
           857: 'throne',
           858: 'tile roof',
           859: 'toaster',
           860: 'tobacco shop, tobacconist shop, tobacconist',
           861: 'toilet seat',
           862: 'torch',
           863: 'totem pole',
           864: 'tow truck, tow car, wrecker',
           865: 'toyshop',
           866: 'tractor',
           867: 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi',
           868: 'tray',
           869: 'trench coat',
           870: 'tricycle, trike, velocipede',
           871: 'trimaran',
           872: 'tripod',
           873: 'triumphal arch',
           874: 'trolleybus, trolley coach, trackless trolley',
           875: 'trombone',
           876: 'tub, vat',
           877: 'turnstile',
           878: 'typewriter keyboard',
           879: 'umbrella',
           880: 'unicycle, monocycle',
           881: 'upright, upright piano',
           882: 'vacuum, vacuum cleaner',
           883: 'vase',
           884: 'vault',
           885: 'velvet',
           886: 'vending machine',
           887: 'vestment',
           888: 'viaduct',
           889: 'violin, fiddle',
           890: 'volleyball',
           891: 'waffle iron',
           892: 'wall clock',
           893: 'wallet, billfold, notecase, pocketbook',
           894: 'wardrobe, closet, press',
           895: 'warplane, military plane',
           896: 'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
           897: 'washer, automatic washer, washing machine',
           898: 'water bottle',
           899: 'water jug',
           900: 'water tower',
           901: 'whiskey jug',
           902: 'whistle',
           903: 'wig',
           904: 'window screen',
           905: 'window shade',
           906: 'Windsor tie',
           907: 'wine bottle',
           908: 'wing',
           909: 'wok',
           910: 'wooden spoon',
           911: 'wool, woolen, woollen',
           912: 'worm fence, snake fence, snake-rail fence, Virginia fence',
           913: 'wreck',
           914: 'yawl',
           915: 'yurt',
           916: 'web site, website, internet site, site',
           917: 'comic book',
           918: 'crossword puzzle, crossword',
           919: 'street sign',
           920: 'traffic light, traffic signal, stoplight',
           921: 'book jacket, dust cover, dust jacket, dust wrapper',
           922: 'menu',
           923: 'plate',
           924: 'guacamole',
           925: 'consomme',
           926: 'hot pot, hotpot',
           927: 'trifle',
           928: 'ice cream, icecream',
           929: 'ice lolly, lolly, lollipop, popsicle',
           930: 'French loaf',
           931: 'bagel, beigel',
           932: 'pretzel',
           933: 'cheeseburger',
           934: 'hotdog, hot dog, red hot',
           935: 'mashed potato',
           936: 'head cabbage',
           937: 'broccoli',
           938: 'cauliflower',
           939: 'zucchini, courgette',
           940: 'spaghetti squash',
           941: 'acorn squash',
           942: 'butternut squash',
           943: 'cucumber, cuke',
           944: 'artichoke, globe artichoke',
           945: 'bell pepper',
           946: 'cardoon',
           947: 'mushroom',
           948: 'Granny Smith',
           949: 'strawberry',
           950: 'orange',
           951: 'lemon',
           952: 'fig',
           953: 'pineapple, ananas',
           954: 'banana',
           955: 'jackfruit, jak, jack',
           956: 'custard apple',
           957: 'pomegranate',
           958: 'hay',
           959: 'carbonara',
           960: 'chocolate sauce, chocolate syrup',
           961: 'dough',
           962: 'meat loaf, meatloaf',
           963: 'pizza, pizza pie',
           964: 'potpie',
           965: 'burrito',
           966: 'red wine',
           967: 'espresso',
           968: 'cup',
           969: 'eggnog',
           970: 'alp',
           971: 'bubble',
           972: 'cliff, drop, drop-off',
           973: 'coral reef',
           974: 'geyser',
           975: 'lakeside, lakeshore',
           976: 'promontory, headland, head, foreland',
           977: 'sandbar, sand bar',
           978: 'seashore, coast, seacoast, sea-coast',
           979: 'valley, vale',
           980: 'volcano',
           981: 'ballplayer, baseball player',
           982: 'groom, bridegroom',
           983: 'scuba diver',
           984: 'rapeseed',
           985: 'daisy',
           986: "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
           987: 'corn',
           988: 'acorn',
           989: 'hip, rose hip, rosehip',
           990: 'buckeye, horse chestnut, conker',
           991: 'coral fungus',
           992: 'agaric',
           993: 'gyromitra',
           994: 'stinkhorn, carrion fungus',
           995: 'earthstar',
           996: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
           997: 'bolete',
           998: 'ear, spike, capitulum',
           999: 'toilet tissue, toilet paper, bathroom tissue'
           }


class ParkCar(data.Dataset):
    """
    停车场景, 车位上车辆再匹配
    """

    def __init__(self,
                 root,
                 list_file,
                 mode='train',
                 input_shape=(1, 128, 128),
                 transform=None):
        """
        数据初始化
        """
        # 读取参数
        self.mode = mode
        self.inp_shape = input_shape
        print('=> input shape: ', self.inp_shape)

        # 数据变换
        self.transform = transform

        # 读取图像路径和标签
        with open(os.path.join(list_file), 'r') as f_h:
            imgs_path_label = f_h.readlines()

        imgs_path_label = [os.path.join(root, img[:-1]) for img in imgs_path_label]
        self.imgs_path_label = np.random.permutation(imgs_path_label)

    def crop_img(self,
                 img,
                 item_name,
                 ratio=0.05):
        """
        根据检测框bbox裁剪输入图像
        :param img:
        :param item_name:
        :param ratio:
        :return:
        """
        H, W = img.shape
        params = item_name.split('_')
        coords = [int(''.join(list(filter(str.isdigit, k))))
                  for k in params]
        x, y, w, h = coords[1], coords[2], coords[3], coords[4]
        if w == 0 or h == 0:
            return img

        # 取ROI
        roi_x_1 = int(x - ratio * float(w))
        roi_x_1 = 0 if roi_x_1 < 0 else roi_x_1
        roi_x_2 = int(x + (1.0 + ratio) * float(w))
        roi_x_2 = W if roi_x_2 >= W else roi_x_2

        roi_y_1 = int(y - ratio * float(h))
        roi_y_1 = 0 if roi_y_1 < 0 else roi_y_1
        roi_y_2 = int(y + (1.0 + ratio) * float(h))
        roi_y_2 = H if roi_y_2 >= H else roi_y_2
        ROI = img[roi_y_1: roi_y_2, roi_x_1: roi_x_2]
        # print(ROI.shape)

        # ----------------- to verify
        # roi_path = '/mnt/diskb/tmp/' + item_name
        # cv2.imwrite(roi_path, ROI)

        return ROI

    def __getitem__(self, idx):
        """
        获取一项数据和标签
        :param idx:
        :return:
        @TODO: padding using pyTorch APIs and transforms
        """
        sample = self.imgs_path_label[idx].split()
        img = Image.open(sample[0])
        img = img.convert('L')  # 转换成32bits灰度

        if self.transform is None:
            # ---------------------padding and resizing img rfb_data
            img = np.array(img)  # H x W x channels

            # crop to reduce the impact of environment...
            # img = self.crop_img(img, os.path.split(sample[0])[-1], ratio=0.05)

            H, W = img.shape
            dim_diff = np.abs(H - W)

            # upper(left) and lower(right) padding
            pad_lu = dim_diff // 2  # integer division
            pad_rd = dim_diff - pad_lu

            # determine padding for each axis: H, W, channels
            pad = ((pad_lu, pad_rd), (0, 0)) if H <= W else (
                (0, 0), (pad_lu, pad_rd))

            # do padding(0.5) and normalize
            padded_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
            img = resize(padded_img,
                         (self.inp_shape[1], self.inp_shape[2], 1),
                         mode='reflect')

            # --------------------- formatting img rfb_data
            # H x W x channels => Channels x H x W
            img = np.transpose(img, (2, 0, 1))

            # turn numpy array to tensor
            img = torch.from_numpy(img).float()

        else:
            img = self.transform(img)

        # --------------------- formatting label rfb_data
        label = np.int32(sample[1])

        return img, label

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path_label)


class VehicleIDGray(data.Dataset):
    """
    VehicleID部分数据用来车辆匹配训练
    """

    def __init__(self,
                 root,
                 mode='train',
                 in_size=(1, 128, 128)):
        """
        :param root:
        :param mode:
        :param in_size:
        """
        if not os.path.isdir(root):
            print('=> [Err]: invalid root.')
            return

        self.in_size = in_size
        print('=> in_size: ', self.in_size)

        if mode == 'train':
            txt_f_path = root + '/train.txt'
        elif mode == 'test':
            txt_f_path = root + '/test.txt'

        # 加载数据和标签
        self.imgs_path, self.labels = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip()
                self.imgs_path.append(root + '/' + line)
                label = int(line.split('/')[0])
                self.labels.append(label)
        print("=> images' path and labels loaded.")

    def __getitem__(self, idx):
        """
        获取样本
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])
        img = img.convert('L')  # 转换成32bits灰度

        # ---------------------padding and resizing img rfb_data
        img = np.array(img)  # H x W x channels

        H, W = img.shape
        dim_diff = np.abs(H - W)

        # upper(left) and lower(right) padding
        pad_lu = dim_diff // 2  # integer division
        pad_rd = dim_diff - pad_lu

        # determine padding for each axis: H, W, channels
        pad = ((pad_lu, pad_rd), (0, 0)) if H <= W else (
            (0, 0), (pad_lu, pad_rd))

        # do padding(0.5) and normalize
        padded_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
        img = resize(padded_img,
                     (self.in_size[1], self.in_size[2], 1),
                     mode='reflect')

        # --------------------- formatting img rfb_data
        # H x W x channels => Channels x H x W
        img = np.transpose(img, (2, 0, 1))

        # turn numpy array to tensor
        img = torch.from_numpy(img).float()

        # --------------------- formatting label rfb_data
        label = np.long(self.labels[idx])

        return img, label

    def __len__(self):
        """
        数据集大小
        :return:
        """
        return len(self.imgs_path)


class VehicleID(data.Dataset):
    """
    VehicleID部分数据用来车辆匹配训练
    """

    def __init__(self,
                 root,
                 mode='train',
                 input_shape=(3, 224, 224)):
        """
        :param root:
        :param mode:
        :param input_shape:
        """
        if not os.path.isdir(root):
            print('=> [Err]: invalid root.')
            return

        self.in_size = input_shape
        print('=> in_size: ', self.in_size)

        if mode == 'train':
            txt_f_path = root + '/train.txt'
        elif mode == 'test':
            txt_f_path = root + '/test.txt'

        # 加载数据和标签
        self.imgs_path, self.labels = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip()
                self.imgs_path.append(root + '/' + line)
                label = int(line.split('/')[0])
                self.labels.append(label)
        print("=> images' path and labels loaded.")

    def pad_resize(self, img):
        """
        :param img: RGB image
        :return:
        """
        img = np.array(img)  # H x W x channels
        H, W, channels = img.shape
        dim_diff = np.abs(H - W)

        # upper(left) and lower(right) padding
        pad_lu = dim_diff // 2  # integer division
        pad_rd = dim_diff - pad_lu

        # determine padding for each axis: H, W, channels
        pad = ((pad_lu, pad_rd), (0, 0), (0, 0)) if H <= W else \
            ((0, 0), (pad_lu, pad_rd), (0, 0))

        # do padding(0.5) and normalize
        padded_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
        img = resize(padded_img,  # H, W, channels
                     (self.in_size[1], self.in_size[2], self.in_size[0]),
                     mode='reflect')
        return img

    def __getitem__(self, idx):
        """
        获取样本
        :param idx:
        :return:
        """
        # 获取样本(包含图像路径和标签)
        img = Image.open(self.imgs_path[idx])

        # ---------------------padding and resizing img rfb_data
        img = self.pad_resize(img)

        # --------------------- formatting img rfb_data
        # H x W x channels => Channels x H x W
        img = np.transpose(img, (2, 0, 1))

        # turn numpy array to tensor
        img = torch.from_numpy(img).float()

        # --------------------- formatting label rfb_data
        label = np.long(self.labels[idx])  # np.int32

        return img, label

    def __len__(self):
        """
        数据集大小
        :return:
        """
        return len(self.imgs_path)


# --------------------------------------
# VehicleID用于MDNet
class VehicleID_All(data.Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 mode='train'):
        """
        :param root:
        :param transforms:
        :param mode:
        """
        if not os.path.isdir(root):
            print('[Err]: invalid root.')
            return

        # 加载图像绝对路径和标签
        if mode == 'train':
            txt_f_path = root + '/attribute/train_all.txt'
        elif mode == 'test':
            txt_f_path = root + '/attribute/test_all.txt'

        if not os.path.isfile(txt_f_path):
            print('=> [Err]: invalid txt file.')
            return

        # 打开vid2TrainID和trainID2Vid映射
        vid2TrainID_path = root + '/attribute/vid2TrainID.pkl'
        trainID2Vid_path = root + '/attribute/trainID2Vid.pkl'
        if not (os.path.isfile(vid2TrainID_path) \
                and os.path.isfile(trainID2Vid_path)):
            print('=> [Err]: invalid vid, train_id mapping file path.')

        with open(vid2TrainID_path, 'rb') as fh_1, \
                open(trainID2Vid_path, 'rb') as fh_2:
            self.vid2TrainID = pickle.load(fh_1)
            self.trainID2Vid = pickle.load(fh_2)

        self.imgs_path, self.lables = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip().split()
                img_path = root + '/image/' + line[0] + '.jpg'
                if os.path.isfile(img_path):
                    self.imgs_path.append(img_path)

                    tr_id = self.vid2TrainID[int(line[3])]
                    label = np.array([int(line[1]),
                                      int(line[2]),
                                      int(tr_id)], dtype=int)
                    self.lables.append(torch.Tensor(label))

        assert len(self.imgs_path) == len(self.lables)
        print('=> total %d samples loaded in %s mode' % (len(self.imgs_path), mode))

        # 加载数据变换
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        """
        关于数据缩放方式: 先默认使用非等比缩放
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # 数据变换, 灰度图转换成'RGB'
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # 图像数据变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.lables[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


# Vehicle ID用于车型和颜色的多标签分类
class VehicleID_MC(data.Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 mode='train'):
        """
        :param root:
        :param transforms:
        :param mode:
        """
        if not os.path.isdir(root):
            print('[Err]: invalid root.')
            return

        # 加载图像绝对路径和标签
        if mode == 'train':
            txt_f_path = root + '/attribute/train.txt'
        elif mode == 'test':
            txt_f_path = root + '/attribute/test.txt'

        if not os.path.isfile(txt_f_path):
            print('=> [Err]: invalid txt file.')
            return

        self.imgs_path, self.lables = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip().split()
                img_path = root + '/image/' + line[0] + '.jpg'
                if os.path.isfile(img_path):
                    self.imgs_path.append(img_path)
                    label = np.array([int(line[1]), int(line[2])], dtype=int)
                    self.lables.append(torch.Tensor(label))

        assert len(self.imgs_path) == len(self.lables)
        print('=> total %d samples loaded in %s mode' % (len(self.imgs_path), mode))

        # 加载数据变换
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        """
        关于数据缩放方式: 先默认使用非等比缩放
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # 数据变换, 灰度图转换成'RGB'
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # 图像数据变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.lables[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


class VehicleReID(data.Dataset):
    """
    通车场景的Vehicle ReID数据
    """

    def __init__(self,
                 root,
                 transform):
        """
        :param root:
        :param transform:
        """
        if not os.path.isdir(root):
            print('=> [Err]: invalid root.')
            return

        # 图像数据路径和label
        for f_name in os.listdir(root):
            f_path = root + '/' + f_name
            if os.path.isdir(f_path):
                print('=> ')

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """

    def __len__(self):
        """
        :return:
        """


class CarMatchData(data.Dataset):
    """
    训练RGB三通道图像
    使用py-torch API进行padding和resize
    """

    def __init__(self,
                 root,
                 list_file,
                 input_shape,
                 transform=None):
        """
        :param root:
        :param list_file:
        :param input_shape:
        :param transform:
        """
        if not os.path.isdir(root):
            print('=> invalid root')
            return

        self.in_size = input_shape
        print('=> in_size: ', self.in_size)

        # 数据变换
        self.transform = transform

        # 读取图像路径和标签
        with open(os.path.join(list_file), 'r') as f_h:
            imgs_path_label = f_h.readlines()

        imgs_path_label = [os.path.join(root, img[:-1]) for img in imgs_path_label]
        self.imgs_path_label = np.random.permutation(imgs_path_label)

    def pad_resize_img(self, img):
        """
        :param img: RGB image
        :return:
        """
        img = np.array(img)  # H x W x channels
        H, W, channels = img.shape
        dim_diff = np.abs(H - W)

        # upper(left) and lower(right) padding
        pad_lu = dim_diff // 2  # integer division
        pad_rd = dim_diff - pad_lu

        # determine padding for each axis: H, W, channels
        pad = ((pad_lu, pad_rd), (0, 0), (0, 0)) if H <= W else \
            ((0, 0), (pad_lu, pad_rd), (0, 0))

        # do padding(0.5) and normalize
        padded_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.0
        img = resize(padded_img,  # H, W, channels
                     (self.in_size[1], self.in_size[2], self.in_size[0]),
                     mode='reflect')
        return img

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        # 获取样本(包含图像路径和标签)
        sample = self.imgs_path_label[idx].split()
        img = Image.open(sample[0])

        if self.transform is None:
            # ---------------------padding and resizing img rfb_data
            img = self.pad_resize_img(img)

            # H x W x channels => Channels x H x W
            img = np.transpose(img, (2, 0, 1))

            # turn numpy array to tensor
            img = torch.from_numpy(img).float()
        else:
            img = self.transform(img)

        # --------------------- formatting label rfb_data
        label = np.long(sample[1])

        return img, label

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path_label)


# ------------------------------------------------
class DataSet(data.Dataset):
    def __init__(self,
                 root,
                 data_list_file,
                 mode='train',
                 transform=None,
                 input_shape=(1, 128, 128)):
        """
        数据集
        """
        # 参数
        self.mode = mode
        self.transforms = transform
        self.input_shape = input_shape

        # 读取图像路径和标签
        with open(os.path.join(data_list_file), 'r') as f_h:
            imgs_path_label = f_h.readlines()

        imgs_path_label = [os.path.join(root, img[:-1]) for img in imgs_path_label]
        self.imgs_path_label = np.random.permutation(imgs_path_label)

        # normalize = T.Normalize(mean=[0.5], std=[0.5])
        normalize = T.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))
        if self.mode == 'train':
            if None == self.transforms:
                self.transforms = T.Compose([
                    T.RandomCrop(self.input_shape[1:]),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, idx):
        sample = self.imgs_path_label[idx]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)

        # 读取灰度图
        data = data.convert('L')
        data = self.transforms(data)
        label = np.int32(splits[1])

        return data.float(), label

    def __len__(self):
        return len(self.imgs_path_label)


def process_webface(root, dst_dir, RATIO=0.2):
    """
    处理WebFace生成训练数据集和验证数据集
    """
    if not os.path.isdir(dst_dir):
        print('[Err]: invalid dst dir.')
        return

    train_webface_f_path = dst_dir + '/' + 'train_webface.txt'
    valid_webface_f_path = dst_dir + '/' + 'valid_webface.txt'

    train_f_h = open(train_webface_f_path, 'w')
    valid_f_h = open(valid_webface_f_path, 'w')

    # 统计类别数
    num_cls = len([x for x in os.listdir(root) if os.path.isdir(str(root + '/' + x))])
    print('=> total %d IDs of person.' % num_cls)

    for id, x in tqdm(enumerate(os.listdir(root))):
        x_path = root + '/' + x
        if os.path.isdir(x_path):
            # 处理每一个目录(ID)
            for y in os.listdir(x_path):
                if y.endswith('.jpg'):
                    jpg_f_path = x_path + '/' + y

                    # 阈值判断
                    if random.random() < RATIO:
                        valid_f_h.write(jpg_f_path + '\t' + str(id) + '\n')
                    else:
                        train_f_h.write(jpg_f_path + '\t' + str(id) + '\n')

    train_f_h.close()
    valid_f_h.close()


def dirs_statistics(root,
                    gt_dirs,
                    le_dirs,
                    limit):
    """
    递归对子目录分类：> limit与否
    :param root:
    :param gt_dirs:
    :param le_dirs:
    :param limit:
    :return:
    """
    if not os.path.isdir(root):
        print('[Err]: invalid root %s' % root)
        return

    for x in os.listdir(root):
        x_path = root + '/' + x

        if os.path.isdir(x_path):
            dirs_statistics(x_path, gt_dirs, le_dirs, limit)
        elif os.path.isfile(x_path):  # 此root是叶子目录
            if len([y for y in os.listdir(root) if y.endswith('.jpg')]) > limit:
                gt_dirs.append(root)
            else:
                le_dirs.append(root)
            break  # 跳出递归


def process_car(root,
                dst_dir,
                limit,
                RATIO=0.1):
    """
    处理每个子目录中car图像>=10
    生成trina_list和valid_list
    :param root:
    :param dst_dir:
    :param RATIO:
    :return:
    """
    if not os.path.isdir(dst_dir):
        print('[Err]: invalid dst dir.')
        return

    train_car_f_path = dst_dir + '/' + 'train_car.txt'
    valid_car_f_path = dst_dir + '/' + 'valid_car.txt'

    train_f_h = open(train_car_f_path, 'w')
    valid_f_h = open(valid_car_f_path, 'w')

    # 获取满足条件的子目录
    gt_dirs = []
    le_dirs = []
    dirs_statistics(root,
                    gt_dirs,
                    le_dirs,
                    limit=limit)

    if len(gt_dirs) == 0:
        print('=> [Warning]: non gt dirs.')
        return
    print('=> total %d sub-dirs meet requirement.' % len(gt_dirs))

    # 处理每一个满足条件的子目录
    train_count, valid_count = 0, 0
    for id, dir in tqdm(enumerate(gt_dirs)):
        for f in os.listdir(dir):
            if f.endswith('.jpg'):
                f_path = dir + '/' + f
                if os.path.isfile(f_path):
                    f_path = f_path.replace(root + '/', '')  # 为了与其他数据形式保持一致
                    if random.random() < RATIO:
                        valid_f_h.write(f_path + '\t' + str(id) + '\n')
                        valid_count += 1
                    else:
                        train_f_h.write(f_path + '\t' + str(id) + '\n')
                        train_count += 1

    print('=> total %d files for training, and total %d files for testing.'
          % (train_count, valid_count))

    train_f_h.close()
    valid_f_h.close()

    print('=> %s generated, total %d train samples.'
          % (train_car_f_path, train_count))
    print('=> %s generated, total %d test samples.'
          % (valid_car_f_path, valid_count))
    return len(gt_dirs)


def gen_data_set(src_root,
                 dst_root):
    """
    for car reid
    :param src_root: '/mnt/diskc/even/CarReID_data'
    :param dst_root:
    :return:
    """
    if not os.path.isdir(src_root):
        print('=> [Err]: invalid dir.')
        return

    # 创建目标根目录
    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)

    # 获取满足条件的子目录
    gt_dirs = []
    le_dirs = []
    dirs_statistics(src_root,
                    gt_dirs,
                    le_dirs,
                    limit=10)
    if len(gt_dirs) == 0:
        print('=> [Warning]: non gt dirs.')
        return
    print('=> total %d sub-dirs meet requirement.' % len(gt_dirs))

    # 将满足条件所有子目录拷贝到
    for dir in gt_dirs:
        dst_dir = '/'.join(dir.split('/')[1:])
        dst_dir = dst_root + '/' \
                  + dst_dir.replace(src_root[1:] + '/', '')
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(dir, dst_dir)
        print('=> %s copied.' % dst_dir)


# ------------------ 统计子目录
def get_sub_dirs(root, dir_list):
    for x in os.listdir(root):
        x_path = root + '/' + x
        if os.path.isdir(x_path):
            get_sub_dirs(x_path, dir_list)
        elif os.path.isfile(x_path):
            parent_dir = os.path.split(x_path)[:-1][0]
            dir_list.append(parent_dir)
            break


from collections import defaultdict


def gen_test_pairs(valid_f_path,
                   dst_dir,
                   num=5000,
                   TH=0.9):
    """
    生成测试pair数据: 一半positive，一半negative
    :param valid_f_path:
    :return:
    """
    if not os.path.isfile(valid_f_path):
        print('[Err]: invalid file.')
        return
    print('=> genarating %d samples...' % num)

    with open(valid_f_path, 'r') as f_h:
        valid_list = f_h.readlines()
        print('=> %s loaded.' % valid_f_path)

        # 字典生成式
        valid_dict = {x.strip().split()[0]: int(x.strip().split()[1]) for x in valid_list}
        inv_dict = defaultdict(list)
        for k, v in valid_dict.items():
            inv_dict[v].append(k)

        # 统计样本数不少于2的id
        big_ids = [k for k, v in inv_dict.items() if len(v) > 1]

    # 添加测试样本
    pair_set = set()
    pos_cnt, neg_cnt, total_cnt = 0, 0, 0

    while total_cnt < num:
        # random value range in [0.0, 1.0]
        rand_val = random.random()

        # anchor and negative
        if rand_val > TH:
            pick_id_1 = random.sample(big_ids, 1)[0]  # 不放回抽取
            pick_id_2 = random.sample(big_ids, 1)[0]  # 不放回抽取
            while pick_id_2 == pick_id_1:
                pick_id_2 = random.sample(big_ids, 1)[0]
            assert pick_id_2 != pick_id_1
            anchor = random.choice(inv_dict[pick_id_1])
            negative = random.choice(inv_dict[pick_id_2])

            # anchor and negative
            pair_set.add(anchor + '\t' + negative + '\t0')

            neg_cnt += 1
            total_cnt += 1

        # anchor and positive
        else:
            pick_id = random.sample(big_ids, 1)[0]  # 不放回抽取

            anchor = random.sample(inv_dict[pick_id], 1)[0]
            positive = random.choice(inv_dict[pick_id])
            while positive == anchor:
                positive = random.choice(inv_dict[pick_id])

            # anchor and positive
            pair_set.add(anchor + '\t' + positive + '\t1')

            pos_cnt += 1
            total_cnt += 1

    print('=> total %d positive pairs and total %d negative pairs.\n'
          % (pos_cnt, neg_cnt))

    print(list(pair_set)[:5])
    print(len(pair_set))

    # 序列化pair_set到dst_dir
    pair_set_f_path = dst_dir + '/' + 'pair_set_car.txt'
    if os.path.isfile(pair_set_f_path):
        os.remove(pair_set_f_path)

    with open(pair_set_f_path, 'w') as f_h:
        for x in pair_set:
            f_h.write(x + '\n')
    print('=> %s generated.' % pair_set_f_path)


if __name__ == '__main__':
    # gen_data_set(src_root='/mnt/diskc/even/CarReID_data',
    #              dst_root='/mnt/diskb/CarReID')

    dir_list = []
    get_sub_dirs(root='/mnt/diskb/CarReIDCrop',
                 dir_list=dir_list)
    print('=> total %d dirs.' % len(dir_list))
    print(dir_list)

    # ------------------------------------
    # dataset = Dataset(root='/rfb_data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
    #                   data_list_file='/rfb_data/Datasets/fv/dataset_v1.1/mix_20w.txt',
    #                   mode='test',
    #                   in_size=(1, 128, 128))

    # trainloader = rfb_data.DataLoader(dataset, batch_size=10)
    # for epoch_i, (rfb_data, label) in enumerate(trainloader):
    #     # imgs_path_label, labels = rfb_data
    #     # print imgs_path_label.numpy().shape
    #     # print rfb_data.cpu().numpy()
    #     # if epoch_i == 0:
    #     img = torchvision.utils.make_grid(rfb_data).numpy()
    #     # print img.shape
    #     # print label.shape
    #     # chw -> hwc
    #     img = np.transpose(img, (1, 2, 0))
    #     # img *= np.array([0.229, 0.224, 0.225])
    #     # img += np.array([0.485, 0.456, 0.406])
    #     img += np.array([1, 1, 1])
    #     img *= 127.5
    #     img = img.astype(np.uint8)
    #     img = img[:, :, [2, 1, 0]]

    #     cv2.imshow('img', img)
    #     cv2.waitKey()
    #     # break
    #     # dst.decode_segmap(labels.numpy()[0], plot=True)
    # ------------------------------------

    # process_webface('/mnt/diskc/even/WebFace',
    #                 dst_dir='/mnt/diskc/even/Car_DR/ArcFace_pytorch/rfb_data')

    # gt_dirs, le_dirs = [], []
    # dirs_statistics(root='/mnt/diskc/even/CarReID_data',
    #                 gt_dirs=gt_dirs,
    #                 le_dirs=le_dirs,
    #                 limit=10)
    # if len(gt_dirs) == 0:
    #     print('=> [Warning]: non gt dirs.')
    # print('=> total %d sub-dirs meet requirement.' % len(gt_dirs))

    # ------------------------------------
    # process_car(root='/mnt/diskc/even/CarReID_data',
    #             dst_dir='/mnt/diskc/even/Car_DR/ArcFace_pytorch/rfb_data')
    #
    # gen_test_pairs(valid_f_path='/mnt/diskc/even/Car_DR/ArcFace_pytorch/rfb_data/valid_car.txt',
    #                dst_dir='/mnt/diskc/even/Car_DR/ArcFace_pytorch/rfb_data')

    # ------------------------------------

    # dataset = CarMatchData(root='/mnt/diskc/even/CarReID_data',
    #                        list_file='/mnt/diskc/even/Car_DR/ArcFace_pytorch/rfb_data/train_car.txt')

    # for i in range(50):
    #     img, label = dataset.__getitem__(i)
    #     print('=> label %d: ' % i, end='')
    #     print(label)

# ------------------------
# ---------- Mixed Difference Network Structure base on vgg16
# class MDNet(torch.nn.Module):
#     def __init__(self,
#                  vgg_orig,
#                  out_ids,
#                  out_attribs):
#         """
#         网络结构定义与初始化
#         @TODO: 使用Sequential包装Conv1, Conv2等
#         @TODO: Sequential用ordereddict显示命名各层
#         """
#         super(MDNet, self).__init__()
#
#         self.out_ids, self.out_attribs = out_ids, out_attribs
#         print('=> out_ids: %d, out_attribs: %d' % (self.out_ids, self.out_attribs))
#
#         feats = vgg_orig.RAModel.features._modules
#         classifier = vgg_orig.RAModel.classifier._modules
#
#         # Conv1
#         self.conv1_1 = copy.deepcopy(feats['0'])  # (0)
#         self.conv1_2 = copy.deepcopy(feats['1'])  # (1)
#         self.conv1_3 = copy.deepcopy(feats['2'])  # (2)
#         self.conv1_4 = copy.deepcopy(feats['3'])  # (3)
#         self.conv1_5 = copy.deepcopy(feats['4'])  # (4)
#
#         self.conv1 = torch.nn.Sequential(
#             self.conv1_1,
#             self.conv1_2,
#             self.conv1_3,
#             self.conv1_4,
#             self.conv1_5
#         )
#
#         # Conv2
#         self.conv2_1 = copy.deepcopy(feats['5'])  # (5)
#         self.conv2_2 = copy.deepcopy(feats['6'])  # (6)
#         self.conv2_3 = copy.deepcopy(feats['7'])  # (7)
#         self.conv2_4 = copy.deepcopy(feats['8'])  # (8)
#         self.conv2_5 = copy.deepcopy(feats['9'])  # (9)
#
#         self.conv2 = torch.nn.Sequential(
#             self.conv2_1,
#             self.conv2_2,
#             self.conv2_3,
#             self.conv2_4,
#             self.conv2_5
#         )
#
#         # Conv3
#         self.conv3_1 = copy.deepcopy(feats['10'])  # (10)
#         self.conv3_2 = copy.deepcopy(feats['11'])  # (11)
#         self.conv3_3 = copy.deepcopy(feats['12'])  # (12)
#         self.conv3_4 = copy.deepcopy(feats['13'])  # (13)
#         self.conv3_5 = copy.deepcopy(feats['14'])  # (14)
#         self.conv3_6 = copy.deepcopy(feats['15'])  # (15)
#         self.conv3_7 = copy.deepcopy(feats['16'])  # (16)
#
#         self.conv3 = torch.nn.Sequential(
#             self.conv3_1,
#             self.conv3_2,
#             self.conv3_3,
#             self.conv3_4,
#             self.conv3_5,
#             self.conv3_6,
#             self.conv3_7
#         )
#
#         # Conv4_1
#         self.conv4_1_1 = copy.deepcopy(feats['17'])  # (17)
#         self.conv4_1_2 = copy.deepcopy(feats['18'])  # (18)
#         self.conv4_1_3 = copy.deepcopy(feats['19'])  # (19)
#         self.conv4_1_4 = copy.deepcopy(feats['20'])  # (20)
#         self.conv4_1_5 = copy.deepcopy(feats['21'])  # (21)
#         self.conv4_1_6 = copy.deepcopy(feats['22'])  # (22)
#         self.conv4_1_7 = copy.deepcopy(feats['23'])  # (23)
#
#         self.conv4_1 = torch.nn.Sequential(
#             self.conv4_1_1,
#             self.conv4_1_2,
#             self.conv4_1_3,
#             self.conv4_1_4,
#             self.conv4_1_5,
#             self.conv4_1_6,
#             self.conv4_1_7
#         )
#
#         # Conv4_2
#         self.conv4_2_1 = copy.deepcopy(self.conv4_1_1)
#         self.conv4_2_2 = copy.deepcopy(self.conv4_1_2)
#         self.conv4_2_3 = copy.deepcopy(self.conv4_1_3)
#         self.conv4_2_4 = copy.deepcopy(self.conv4_1_4)
#         self.conv4_2_5 = copy.deepcopy(self.conv4_1_5)
#         self.conv4_2_6 = copy.deepcopy(self.conv4_1_6)
#         self.conv4_2_7 = copy.deepcopy(self.conv4_1_7)
#
#         self.conv4_2 = torch.nn.Sequential(
#             self.conv4_2_1,
#             self.conv4_2_2,
#             self.conv4_2_3,
#             self.conv4_2_4,
#             self.conv4_2_5,
#             self.conv4_2_6,
#             self.conv4_2_7
#         )
#
#         # Conv5_1
#         self.conv5_1_1 = copy.deepcopy(feats['24'])  # (24)
#         self.conv5_1_2 = copy.deepcopy(feats['25'])  # (25)
#         self.conv5_1_3 = copy.deepcopy(feats['26'])  # (26)
#         self.conv5_1_4 = copy.deepcopy(feats['27'])  # (27)
#         self.conv5_1_5 = copy.deepcopy(feats['28'])  # (28)
#         self.conv5_1_6 = copy.deepcopy(feats['29'])  # (29)
#         self.conv5_1_7 = copy.deepcopy(feats['30'])  # (30)
#
#         self.conv5_1 = torch.nn.Sequential(
#             self.conv5_1_1,
#             self.conv5_1_2,
#             self.conv5_1_3,
#             self.conv5_1_4,
#             self.conv5_1_5,
#             self.conv5_1_6,
#             self.conv5_1_7
#         )
#
#         # Conv5_2
#         self.conv5_2_1 = copy.deepcopy(self.conv5_1_1)
#         self.conv5_2_2 = copy.deepcopy(self.conv5_1_2)
#         self.conv5_2_3 = copy.deepcopy(self.conv5_1_3)
#         self.conv5_2_4 = copy.deepcopy(self.conv5_1_4)
#         self.conv5_2_5 = copy.deepcopy(self.conv5_1_5)
#         self.conv5_2_6 = copy.deepcopy(self.conv5_1_6)
#         self.conv5_2_7 = copy.deepcopy(self.conv5_1_7)
#
#         self.conv5_2 = torch.nn.Sequential(
#             self.conv5_2_1,
#             self.conv5_2_2,
#             self.conv5_2_3,
#             self.conv5_2_4,
#             self.conv5_2_5,
#             self.conv5_2_6,
#             self.conv5_2_7
#         )
#
#         # FC6_1
#         self.FC6_1_1 = copy.deepcopy(classifier['0'])  # (0)
#         self.FC6_1_2 = copy.deepcopy(classifier['1'])  # (1)
#         self.FC6_1_3 = copy.deepcopy(classifier['2'])  # (2)
#         self.FC6_1_4 = copy.deepcopy(classifier['3'])  # (3)
#         self.FC6_1_5 = copy.deepcopy(classifier['4'])  # (4)
#         self.FC6_1_6 = copy.deepcopy(classifier['5'])  # (5)
#
#         self.FC6_1 = torch.nn.Sequential(
#             self.FC6_1_1,
#             self.FC6_1_2,
#             self.FC6_1_3,
#             self.FC6_1_4,
#             self.FC6_1_5,
#             self.FC6_1_6
#         )
#
#         # FC6_2
#         self.FC6_2_1 = copy.deepcopy(self.FC6_1_1)
#         self.FC6_2_2 = copy.deepcopy(self.FC6_1_2)
#         self.FC6_2_3 = copy.deepcopy(self.FC6_1_3)
#         self.FC6_2_4 = copy.deepcopy(self.FC6_1_4)
#         self.FC6_2_5 = copy.deepcopy(self.FC6_1_5)
#         self.FC6_2_6 = copy.deepcopy(self.FC6_1_6)
#
#         self.FC6_2 = torch.nn.Sequential(
#             self.FC6_2_1,
#             self.FC6_2_2,
#             self.FC6_2_3,
#             self.FC6_2_4,
#             self.FC6_2_5,
#             self.FC6_2_6
#         )
#
#         # FC7_1
#         self.FC7_1 = copy.deepcopy(classifier['6'])  # (6): 4096, 1000
#
#         # FC7_2
#         self.FC7_2 = copy.deepcopy(self.FC7_1)
#
#         # ------------------------------ extra layers: FC8 and FC9
#         self.FC_8 = torch.nn.Linear(in_features=2000,  # 2048
#                                     out_features=1024)  # 1024
#
#         # final output layer: vehicle id classifier: using arc_fc_br2
#         # self.FC_9 = torch.nn.Linear(in_features=1000,
#         #                             out_features=out_ids)
#
#         # attribute classifiers: out_attribs to be decided
#         self.attrib_classifier = torch.nn.Linear(in_features=1000,
#                                                  out_features=out_attribs)
#
#         # Arc FC layer for branch_2 and branch_3
#         self.arc_fc_br2 = ArcFC(in_features=1000,
#                                 out_features=out_ids,
#                                 s=30.0,
#                                 m=0.5,
#                                 easy_margin=False)
#         self.arc_fc_br3 = ArcFC(in_features=1024,
#                                 out_features=out_ids,
#                                 s=30.0,
#                                 m=0.5,
#                                 easy_margin=False)
#
#         # 构建分支
#         self.shared_layers = torch.nn.Sequential(
#             self.conv1,
#             self.conv2,
#             self.conv3
#         )
#
#         self.branch_1_feats = torch.nn.Sequential(
#             self.shared_layers,
#             self.conv4_1,
#             self.conv5_1,
#         )
#
#         self.branch_1_fc = torch.nn.Sequential(
#             self.FC6_1,
#             self.FC7_1
#         )
#
#         self.branch_1 = torch.nn.Sequential(
#             self.branch_1_feats,
#             self.branch_1_fc
#         )
#
#         self.branch_2_feats = torch.nn.Sequential(
#             self.shared_layers,
#             self.conv4_2,
#             self.conv5_2
#         )
#
#         self.branch_2_fc = torch.nn.Sequential(
#             self.FC6_2,
#             self.FC7_2
#         )
#
#         self.branch_2 = torch.nn.Sequential(
#             self.branch_2_feats,
#             self.branch_2_fc
#         )
#
#     def forward(self,
#                 X,
#                 branch,
#                 label=None):
#         """
#         先单独训练branch_1, 然后brach_1, branch_2, branch_3联合训练
#         :param X:
#         :param branch:
#         :param label:
#         :return:
#         """
#         # batch size
#         N = X.size(0)
#
#         if branch == 1:  # train attributes classification
#             X = self.branch_1_feats(X)
#
#             # reshape and connect to FC layers
#             X = X.view(N, -1)
#             X = self.branch_1_fc(X)
#
#             assert X.size() == (N, 1000)
#
#             X = self.attrib_classifier(X)
#
#             assert X.size() == (N, self.out_attribs)
#
#             return X
#
#         elif branch == 2:  # get vehicle fine-grained feature
#             if label is None:
#                 print('=> label is None.')
#                 return None
#             X = self.branch_2_feats(X)
#
#             # reshape and connect to FC layers
#             X = X.view(N, -1)
#             X = self.branch_2_fc(X)
#
#             assert X.size() == (N, 1000)
#
#             X = self.arc_fc_br2.forward(input=X, label=label)
#
#             assert X.size() == (N, self.out_ids)
#
#             return X
#
#         elif branch == 3:  # overall: combine branch_1 and branch_2
#             if label is None:
#                 print('=> label is None.')
#                 return None
#             branch_1 = self.branch_1_feats(X)
#             branch_2 = self.branch_2_feats(X)
#
#             # reshape and connect to FC layers
#             branch_1 = branch_1.view(N, -1)
#             branch_2 = branch_2.view(N, -1)
#             branch_1 = self.branch_1_fc(branch_1)
#             branch_2 = self.branch_2_fc(branch_2)
#
#             assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)
#
#             fusion_feats = torch.cat((branch_1, branch_2), dim=1)
#
#             assert fusion_feats.size() == (N, 2000)
#
#             # connect to FC8: output 1024 dim feature vector
#             X = self.FC_8(fusion_feats)
#
#             # connect to classifier: arc_fc_br3
#             X = self.arc_fc_br3.forward(input=X, label=label)
#
#             assert X.size() == (N, self.out_ids)
#
#             return X
#
#         elif branch == 4:  # test pre-trained weights
#             # extract features
#             X = self.branch_1_feats(X)
#
#             # flatten and connect to FC layers
#             X = X.view(N, -1)
#             X = self.branch_1_fc(X)
#
#             assert X.size() == (N, 1000)
#
#             return X
#
#         elif branch == 5:
#             # 前向运算提取用于Vehicle ID的特征向量
#             branch_1 = self.branch_1_feats(X)
#             branch_2 = self.branch_2_feats(X)
#
#             # reshape and connect to FC layers
#             branch_1 = branch_1.view(N, -1)
#             branch_2 = branch_2.view(N, -1)
#             branch_1 = self.branch_1_fc(branch_1)
#             branch_2 = self.branch_2_fc(branch_2)
#
#             assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)
#
#             # feature fusion
#             fusion_feats = torch.cat((branch_1, branch_2), dim=1)
#
#             assert fusion_feats.size() == (N, 2000)
#
#             # connect to FC8: output 1024 dim feature vector
#             X = self.FC_8(fusion_feats)
#
#             assert X.size() == (N, 1024)
#
#             return X
#
#         else:
#             print('=> invalid branch')
#             return None
