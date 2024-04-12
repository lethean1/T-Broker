import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
err_o = {
    "2_6_1500_42_T5": [
        0.04293668631762982,
        0.04301151976444593,
        0.021547705502392333,
        0.03650798519506815,
        0.06175881120721392,
        0.050421056486934,
        0.09821252383143175,
        0.12305203730217149,
        0.1176115365476628,
        0.06171426781376503,
    ],
    "2_7_1500_42_T5": [
        0.0052228043180325345,
        -0.015345704975124547,
        0.03937070385806809,
        0.031253937247723816,
        0.013584978278419185,
        0.0713567666666667,
        0.025692838139491,
        0.040044149901436156,
        0.11919856252698778,
        0.10750171164930072,
        0.1330891526706091,
        0.07289946161644599,
        -0.018873891936543652,
        -0.03623176296817803,
    ],
    "2_16_8000_42_ROBERTA": [
        0.022832638354291983,
        0.013797521769637168,
        0.011126229114269918,
        0.007400208957535811,
        0.0068614106433385865,
        0.0006498437306359131,
        0.00035757445780923826,
        0.004245260552214248,
        0.002940446528157291,
        0.0029221466466191315,
        0.007350572170584923,
    ],
    "2_32_8000_42_ROBERTA": [
        -0.04180953174616839,
        -0.021392718968486234,
        -0.044680201214052126,
        -0.038017631711727264,
        -0.04750202388496642,
        -0.04421753022214566,
        -0.05061083120371964,
        -0.04174244324952639,
    ],
    "2_40_8000_42_ROBERTA": [
        -0.06295386452504607,
        -0.06333439298877536,
        -0.061632289135533536,
        -0.0669194456022784,
        -0.05905185697771816,
        -0.06701280478304569,
        -0.070016468805495,
        -0.06419264716870489,
    ],
    "2_20_5000_42_qa": [
        -0.05537056189262504,
        -0.06699886498396777,
        -0.06781096778755234,
        -0.06495225620323934,
        -0.06871066952232184,
        -0.06725417601742979,
        -0.07046474593439124,
        -0.07326917753021453,
        -0.06377656213927481,
        -0.06676189269094786,
        -0.06660246786154728,
        -0.07018940264737311,
    ],
    "2_30_5000_42_qa": [
        -0.0956577462409488,
        -0.09550397077491916,
        -0.09088994783546443,
        -0.10040765638576483,
        -0.09411276580650131,
        -0.09444702218456327,
        -0.09803709844399935,
    ],
    "2_8_25000_42_TC": [0.281415844509564],
    "2_16_25000_42_TC": [
        0.13893257135844247,
        0.11616454540837894,
        0.1505455593906051,
        0.10308866632247152,
        0.12339726298772735,
        0.08724975790097328,
        0.09811280067710543,
        0.0907988510706729,
    ],
    "2_32_25000_42_TC": [
        0.02651359159256971,
        0.034714958975261924,
        0.026204550682278276,
        0.03407391081961441,
        0.030246780271150568,
        0.043494885051500926,
        0.05353937615987318,
        0.02605255209085302,
        0.032879581177920496,
        0.03194931714059331,
        0.0311449139184787,
    ],
    "2_40_25000_42_TC": [
        0.02110893074134808,
        0.0031428679123155293,
        0.01570160403037885,
        0.007106798662293799,
        0.00806393319237074,
        0.023774235833261416,
        0.0051335516268231135,
        0.005355472348321317,
        0.007689708820229546,
        0.010727555070337406,
    ],
}

err_o = {'2_4_1500_42_T5': [0.05550756905124284, 0.04952204694740432, 0.03631536941484859, 0.023521640098088475, 0.0014056124048703125, 0.017959318078809367, 0.03392702622188419, 0.03572898955690845, 0.03132224515474365, 0.036605520387282306], '2_6_1500_42_T5': [0.051132050892528404, 0.04128080603606914, 0.020209294562016852, 0.009291793706293447, 0.0041717880106737605, 0.015369409440559492, 0.034515471181450215, 0.03264310785793168, 0.034695498214942964, 0.04602426035149073], '2_7_1500_42_T5': [0.07557765697925457, 0.06050270902093297, 0.05172933160612016, 0.04265955737350935, 0.024780474082417916, 0.0011001333896557676, 0.013289519290340448, 0.02704375093400902, 0.012410379639538503, 0.022725951713132457, 0.018737251187458934, 0.03426544681310429, 0.05797044996714548, 0.05588333217872887], '2_16_8000_42_ROBERTA': [0.05971822481319476, 0.05321618990340794, 0.044196581939128704, 0.04244613088208485, 0.039690577300892864, 0.03312961502642595, 0.03367636188263155, 0.03786808778020771, 0.03914383044468727, 0.035134353499179725, 0.03978170177692718], '2_32_8000_42_ROBERTA': [0.017608954787325756, 0.03861808170311702, 0.0139926296624763, 0.02553042887032892, 0.009859686662648595, 0.013389908808334806, 0.012270570079214817, 0.023980575245393564], '2_40_8000_42_ROBERTA': [0.010632321594906995, 0.0056901202211425165, 0.009208297470262986, 8.97424024124994e-05, 0.009794660345116458, 0.004349862221477609, 0.0036079196515329693, 0.00493622509633108], '2_10_5000_42_qa': [0.002148083246788583, 0.0022745430992075675, 0.010475784913910819, 0.015530457729798873], '2_20_5000_42_qa': [0.032756293940639704, 0.022479224138781496, 0.02659005205952485, 0.032838510499054493, 0.03102974621392748, 0.017957313425963958, 0.02025937706158018, 0.02987871439611937, 0.021985924788292404, 0.019601644594261274, 0.026836701734769455, 0.032838510499054493], '2_30_5000_42_qa': [0.03049249201971952, 0.014470307456478232, 0.02086377533507925, 0.022327340271144556, 0.035037246294869755, 0.03819546536743174, 0.046976854983823674, 0.03626972203050369], '2_8_25000_42_TC': [0.03718954048431763], '2_16_25000_42_TC': [0.007913343021582898, 0.010831174363097763, 0.00862160481591195, 0.011166438391874789, 0.012676023038510465, 0.010475430571307714, 0.011152535945831609, 0.010513004164198132], '2_32_25000_42_TC': [0.0023397720485958764, 0.0055834606303371784, 0.00929052265164201, 0.0020234011444669243, 0.004681807949643424, 0.0007600513513513717, 0.004369524016198599, 0.015963510705167716, 0.009778053129676986, 0.010658412316225047], '2_40_25000_42_TC': [0.015580120376283773, 0.014112958902217995, 0.010729861284197902, 0.02272606392508856, 0.017519611512902433, 0.011823572848882441, 0.00724948119444211, 8.632617588665677e-05, 0.015966146422715218, 0.010615367342711537]}
err_o = {'6.0_2_4_1500_42_T5': 0.059496719855011065, '6.5_2_4_1500_42_T5': 0.05488942040252265, '7.0_2_4_1500_42_T5': 0.04476994228207347, '7.5_2_4_1500_42_T5': 0.03456403462591542, '8.0_2_4_1500_42_T5': 0.015705307687604957, '8.5_2_4_1500_42_T5': 0.0018469894244377092, '9.0_2_4_1500_42_T5': 0.017125675754153684, '9.5_2_4_1500_42_T5': 0.01989222191570869, '10.0_2_4_1500_42_T5': 0.015578440362169377, '10.5_2_4_1500_42_T5': 0.021067173199792764, '8.0_2_6_1500_42_T5': 0.06656446641701869, '8.5_2_6_1500_42_T5': 0.0588310547532509, '9.0_2_6_1500_42_T5': 0.041281483194180305, '9.5_2_6_1500_42_T5': 0.03192442586785164, '10.0_2_6_1500_42_T5': 0.02001267794757953, '10.5_2_6_1500_42_T5': 0.009767982242828618, '11.0_2_6_1500_42_T5': 0.008461992182786114, '11.5_2_6_1500_42_T5': 0.006826627290927427, '12.0_2_6_1500_42_T5': 0.009003902830974204, '12.5_2_6_1500_42_T5': 0.02069898334272313, '8.0_2_7_1500_42_T5': 0.0858944693820447, '8.5_2_7_1500_42_T5': 0.07426137631236772, '9.0_2_7_1500_42_T5': 0.06743918678997114, '9.5_2_7_1500_42_T5': 0.060179379928494016, '10.0_2_7_1500_42_T5': 0.04520738561498678, '10.5_2_7_1500_42_T5': 0.02245318719602529, '11.0_2_7_1500_42_T5': 0.0354681386879585, '11.5_2_7_1500_42_T5': 0.047708166415611014, '12.0_2_7_1500_42_T5': 0.012217532157177115, '12.5_2_7_1500_42_T5': 0.002508575886851516, '13.0_2_7_1500_42_T5': 0.006473277697307293, '13.5_2_7_1500_42_T5': 0.009060368001514761, '14.0_2_7_1500_42_T5': 0.033792728364871405, '14.5_2_7_1500_42_T5': 0.03156773506342588, '7.0_2_16_8000_42_ROBERTA': 0.06386570730834747, '7.5_2_16_8000_42_ROBERTA': 0.058169430909714374, '8.0_2_16_8000_42_ROBERTA': 0.050147026819558495, '8.5_2_16_8000_42_ROBERTA': 0.04856865583008228, '9.0_2_16_8000_42_ROBERTA': 0.04607093184979123, '9.5_2_16_8000_42_ROBERTA': 0.04006298110566816, '10.0_2_16_8000_42_ROBERTA': 0.04056653261059614, '10.5_2_16_8000_42_ROBERTA': 0.044409613375130566, '11.0_2_16_8000_42_ROBERTA': 0.045573143155331196, '11.5_2_16_8000_42_ROBERTA': 0.04190675746464103, '12.0_2_16_8000_42_ROBERTA': 0.04615384615384605, '23.0_2_32_8000_42_ROBERTA': 0.025752873081117417, '23.5_2_32_8000_42_ROBERTA': 0.04529387587340736, '24.0_2_32_8000_42_ROBERTA': 0.022308275107332198, '24.5_2_32_8000_42_ROBERTA': 0.03321401814700737, '23.0_2_40_8000_42_ROBERTA': 0.024115098504046455, '23.5_2_40_8000_42_ROBERTA': 0.019385575817315585, '24.0_2_40_8000_42_ROBERTA': 0.022757039947609702, '24.5_2_40_8000_42_ROBERTA': 0.013795952085914926, '25.0_2_40_8000_42_ROBERTA': 0.02331669802830736, '25.5_2_40_8000_42_ROBERTA': 0.01809508142786645, '26.0_2_40_8000_42_ROBERTA': 0.010362264776589572, '26.5_2_40_8000_42_ROBERTA': 0.018660090423345748, '18.0_2_32_8000_42_ROBERTA': 0.014411739016507916, '18.5_2_32_8000_42_ROBERTA': 0.01609798775153109, '19.0_2_32_8000_42_ROBERTA': 0.010176567800295744, '19.5_2_32_8000_42_ROBERTA': 0.010440229685053096, '20.0_2_32_8000_42_ROBERTA': 0.012642776179265873, '7.0_2_10_5000_42_qa': 0.01092092880322714, '7.5_2_10_5000_42_qa': 0.006613338574007128, '8.0_2_10_5000_42_qa': 0.019025351713062057, '8.5_2_10_5000_42_qa': 0.006619113305232457, '23.0_2_20_5000_42_qa': 0.024166385988548374, '23.5_2_20_5000_42_qa': 0.013498875093742107, '24.0_2_20_5000_42_qa': 0.01773910133043264, '24.5_2_20_5000_42_qa': 0.02425263157894733, '25.0_2_20_5000_42_qa': 0.022358577792720824, '25.5_2_20_5000_42_qa': 0.008875248838752431, '26.0_2_20_5000_42_qa': 0.011223811107415983, '26.5_2_20_5000_42_qa': 0.021156913777180725, '27.0_2_20_5000_42_qa': 0.012992421087698863, '27.5_2_20_5000_42_qa': 0.010551678298437986, '28.0_2_20_5000_42_qa': 0.017994643454971593, '28.5_2_20_5000_42_qa': 0.02425263157894733, '25.0_2_30_5000_42_qa': 0.013031603589543406, '25.5_2_30_5000_42_qa': 0.003148276126852466, '26.0_2_30_5000_42_qa': 0.0032457496136011395, '26.5_2_30_5000_42_qa': 0.00472099682687087, '27.0_2_30_5000_42_qa': 0.01771715271244897, '27.5_2_30_5000_42_qa': 0.02099882029099479, '28.0_2_30_5000_42_qa': 0.0302356955797158, '28.5_2_30_5000_42_qa': 0.018995290423861753, '4.0_2_8_25000_42_TC': 0.032886951105574595, '4.0_2_16_25000_42_TC': 0.0026911431640789323, '4.5_2_16_25000_42_TC': 0.015354416361670268, '5.0_2_16_25000_42_TC': 0.003260330078355299, '5.5_2_16_25000_42_TC': 0.0014881108297459154, '6.0_2_16_25000_42_TC': 0.001670512819647316, '6.5_2_16_25000_42_TC': 0.010438719876996716, '7.0_2_16_25000_42_TC': 0.0036658827082979737, '7.5_2_16_25000_42_TC': 0.00824599433218747, '10.5_2_32_25000_42_TC': 0.00851913943465359, '11.0_2_32_25000_42_TC': 0.0006723862293351111, '11.5_2_32_25000_42_TC': 0.0030427460091825265, '12.0_2_32_25000_42_TC': 0.004215032599929795, '12.5_2_32_25000_42_TC': 0.00157217867815075, '13.0_2_32_25000_42_TC': 0.005466247784976368, '13.5_2_32_25000_42_TC': 0.0018834445303576506, '14.0_2_32_25000_42_TC': 0.00980181461463239, '14.5_2_32_25000_42_TC': 0.0035338810849015433, '15.0_2_32_25000_42_TC': 0.004421257405606154, '15.0_2_40_25000_42_TC': 0.008659049608491462, '15.5_2_40_25000_42_TC': 0.007169413274259465, '16.0_2_40_25000_42_TC': 0.003753219343381883, '16.5_2_40_25000_42_TC': 0.015986599219640553, '17.0_2_40_25000_42_TC': 0.010641081552551232, '17.5_2_40_25000_42_TC': 0.004856473853091686, '18.0_2_40_25000_42_TC': 0.000258977900552496, '18.5_2_40_25000_42_TC': 0.007027165995372296, '19.0_2_40_25000_42_TC': 0.009056866672472404, '19.5_2_40_25000_42_TC': 0.017301331524043694}
err = {
  "6.0_2_4_1500_42_T5": 0.059496719855011065,
  "6.5_2_4_1500_42_T5": 0.05488942040252265,
  "7.0_2_4_1500_42_T5": 0.04476994228207347,
  "7.5_2_4_1500_42_T5": 0.03456403462591542,
  "8.0_2_4_1500_42_T5": 0.015705307687604957,
  "8.5_2_4_1500_42_T5": 0.0018469894244377092,
  "9.0_2_4_1500_42_T5": 0.017125675754153684,
  "9.5_2_4_1500_42_T5": 0.01989222191570869,
  "10.0_2_4_1500_42_T5": 0.015578440362169377,
  "10.5_2_4_1500_42_T5": 0.021067173199792764,
  "8.0_2_6_1500_42_T5": 0.06656446641701869,
  "8.5_2_6_1500_42_T5": 0.0588310547532509,
  "9.0_2_6_1500_42_T5": 0.041281483194180305,
  "9.5_2_6_1500_42_T5": 0.03192442586785164,
  "10.0_2_6_1500_42_T5": 0.02001267794757953,
  "10.5_2_6_1500_42_T5": 0.009767982242828618,
  "11.0_2_6_1500_42_T5": 0.008461992182786114,
  "11.5_2_6_1500_42_T5": 0.006826627290927427,
  "12.0_2_6_1500_42_T5": 0.009003902830974204,
  "12.5_2_6_1500_42_T5": 0.02069898334272313,
  "8.0_2_7_1500_42_T5": 0.0858944693820447,
  "8.5_2_7_1500_42_T5": 0.07426137631236772,
  "9.0_2_7_1500_42_T5": 0.06743918678997114,
  "9.5_2_7_1500_42_T5": 0.060179379928494016,
  "10.0_2_7_1500_42_T5": 0.04520738561498678,
  "10.5_2_7_1500_42_T5": 0.02245318719602529,
  "11.0_2_7_1500_42_T5": 0.0354681386879585,
  "11.5_2_7_1500_42_T5": 0.047708166415611014,
  "12.0_2_7_1500_42_T5": 0.012217532157177115,
  "12.5_2_7_1500_42_T5": 0.002508575886851516,
  "13.0_2_7_1500_42_T5": 0.006473277697307293,
  "13.5_2_7_1500_42_T5": 0.009060368001514761,
  "14.0_2_7_1500_42_T5": 0.033792728364871405,
  "14.5_2_7_1500_42_T5": 0.03156773506342588,
  "7.0_2_16_8000_42_ROBERTA": 0.005141448344613214,
  "7.5_2_16_8000_42_ROBERTA": 0.002790754728866315,
  "8.0_2_16_8000_42_ROBERTA": 0.006100568805573471,
  "8.5_2_16_8000_42_ROBERTA": 0.0026274945185455413,
  "9.0_2_16_8000_42_ROBERTA": 0.005405405405405436,
  "9.5_2_16_8000_42_ROBERTA": 0.004853035436315366,
  "10.0_2_16_8000_42_ROBERTA": 0.00595838298652494,
  "10.5_2_16_8000_42_ROBERTA": 0.0020914794944074727,
  "11.0_2_16_8000_42_ROBERTA": 0.007237199203907934,
  "11.5_2_16_8000_42_ROBERTA": 9.1132780461308e-05,
  "12.0_2_16_8000_42_ROBERTA": 0.006142844045108792,
  "23.0_2_32_8000_42_ROBERTA": 0.007023324373536827,
  "23.5_2_32_8000_42_ROBERTA": 0.009561891515994512,
  "24.0_2_32_8000_42_ROBERTA": 0.007110648629899475,
  "24.5_2_32_8000_42_ROBERTA": 0.010879972147271303,
  "23.0_2_40_8000_42_ROBERTA": 0.016345990124297525,
  "23.5_2_40_8000_42_ROBERTA": 0.009812214515310409,
  "24.0_2_40_8000_42_ROBERTA": 0.007341152645346423,
  "24.5_2_40_8000_42_ROBERTA": 0.018166311300639622,
  "25.0_2_40_8000_42_ROBERTA": 0.013154544683017882,
  "25.5_2_40_8000_42_ROBERTA": 0.011352084039308739,
  "26.0_2_40_8000_42_ROBERTA": 0.01955760526091034,
  "26.5_2_40_8000_42_ROBERTA": 0.016692215976835238,
  "18.0_2_32_8000_42_ROBERTA": 0.014411739016507916,
  "18.5_2_32_8000_42_ROBERTA": 0.01609798775153109,
  "19.0_2_32_8000_42_ROBERTA": 0.010176567800295744,
  "19.5_2_32_8000_42_ROBERTA": 0.010440229685053096,
  "20.0_2_32_8000_42_ROBERTA": 0.012642776179265873,
  "7.0_2_10_5000_42_qa": 0.01092092880322714,
  "7.5_2_10_5000_42_qa": 0.006613338574007128,
  "8.0_2_10_5000_42_qa": 0.019025351713062057,
  "8.5_2_10_5000_42_qa": 0.006619113305232457,
  "23.0_2_20_5000_42_qa": 0.024166385988548374,
  "23.5_2_20_5000_42_qa": 0.013498875093742107,
  "24.0_2_20_5000_42_qa": 0.01773910133043264,
  "24.5_2_20_5000_42_qa": 0.02425263157894733,
  "25.0_2_20_5000_42_qa": 0.022358577792720824,
  "25.5_2_20_5000_42_qa": 0.008875248838752431,
  "26.0_2_20_5000_42_qa": 0.011223811107415983,
  "26.5_2_20_5000_42_qa": 0.021156913777180725,
  "27.0_2_20_5000_42_qa": 0.012992421087698863,
  "27.5_2_20_5000_42_qa": 0.010551678298437986,
  "28.0_2_20_5000_42_qa": 0.017994643454971593,
  "28.5_2_20_5000_42_qa": 0.02425263157894733,
  "25.0_2_30_5000_42_qa": 0.013031603589543406,
  "25.5_2_30_5000_42_qa": 0.003148276126852466,
  "26.0_2_30_5000_42_qa": 0.0032457496136011395,
  "26.5_2_30_5000_42_qa": 0.00472099682687087,
  "27.0_2_30_5000_42_qa": 0.01771715271244897,
  "27.5_2_30_5000_42_qa": 0.02099882029099479,
  "28.0_2_30_5000_42_qa": 0.0302356955797158,
  "28.5_2_30_5000_42_qa": 0.018995290423861753,
  "4.0_2_8_25000_42_TC": 0.032886951105574595,
  "4.0_2_16_25000_42_TC": 0.0026911431640789323,
  "4.5_2_16_25000_42_TC": 0.015354416361670268,
  "5.0_2_16_25000_42_TC": 0.003260330078355299,
  "5.5_2_16_25000_42_TC": 0.0014881108297459154,
  "6.0_2_16_25000_42_TC": 0.001670512819647316,
  "6.5_2_16_25000_42_TC": 0.010438719876996716,
  "7.0_2_16_25000_42_TC": 0.0036658827082979737,
  "7.5_2_16_25000_42_TC": 0.00824599433218747,
  "10.0_2_32_25000_42_TC": 0.013461706187624675,
  "10.5_2_32_25000_42_TC": 0.008692112369155534,
  "11.0_2_32_25000_42_TC": 0.010246157132903529,
  "11.5_2_32_25000_42_TC": 0.012820460436218236,
  "12.0_2_32_25000_42_TC": 0.004215032599929795,
  "12.5_2_32_25000_42_TC": 0.007330325847605635,
  "13.0_2_32_25000_42_TC": 0.007117864015383258,
  "13.5_2_32_25000_42_TC": 0.0013569654681318852,
  "14.0_2_32_25000_42_TC": 0.0057539945207876865,
  "14.5_2_32_25000_42_TC": 0.013290479499652546,
  "15.0_2_32_25000_42_TC": 0.008640251352766581,
  "15.0_2_40_25000_42_TC": 0.0005633137309053416,
  "15.5_2_40_25000_42_TC": 0.0026400688575630117,
  "16.0_2_40_25000_42_TC": 0.003058099385387743,
  "16.5_2_40_25000_42_TC": 0.00435408454201747,
  "17.0_2_40_25000_42_TC": 0.005815972222222237,
  "17.5_2_40_25000_42_TC": 8.629616845004233e-05,
  "18.0_2_40_25000_42_TC": 0.0006909059504276061,
  "18.5_2_40_25000_42_TC": 0.006427516720229385,
  "19.0_2_40_25000_42_TC": 0.010112457501525556,
  "19.5_2_40_25000_42_TC": 0.0038118340119552775
}
err = err.values()
err = list(err)[:110]
err.sort()
print(err[98])
print(err[104])
print(len(err))
from math import ceil
err = [abs(x) for x in err]

err.sort()
dx = 0.001
x = [i * dx for i in range(int(0.5 / dx))]
m = {True: 1, False: 0}

cdf = [sum([m[u < i] for u in err]) / len(err) for i in x]

def smooth_xy(lx, ly):
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]
print(x)
print(cdf)

f1 = True 
f2 = True 
x1 = 0
x2 = 0
for i, j in zip(x, cdf):
    if j > 0.9 and f1:
        print(i)
        f1 = False
        x1 = i*100
    if j > 0.95 and f2:
        print(i)
        f2 = False
        x2 = i*100
x1 = 0.041281483194180305 * 100
x2 = 0.059496719855011065 * 100


fontsize = 15
family = 'Times New Roman'
font1 = {'family' : family,
'weight' : 'normal',
'size' : fontsize,
}
x = [v * 100 for v in x]
plt.figure(figsize=(6, 2))
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
plt.rcParams['hatch.linewidth'] = 1
lns = plt.plot(x, cdf)
plt.xlim(0, 10)
plt.ylim(0, 1)
plt.xlabel("Error(%)", fontdict=font1)
plt.ylabel("CDF", fontdict=font1)
y1 = 0.895
y2 = 0.95
plt.plot([x1, x1], [0, 0.9], 'k--', linewidth=1)
plt.plot([x2, x2], [0, y2], 'k--', linewidth=1)
plt.plot([0, x1], [y1, y1], 'k--', linewidth=1)
plt.plot([0, x2], [y2, y2], 'k--', linewidth=1)
plt.text(7, 0.7, f'90% < {x1:.1f}%', fontdict=font1)
plt.text(7, 0.5, f'95% < {x2:.1f}%', fontdict=font1)
plt.xticks(family=family, fontsize=fontsize)
plt.yticks(family=family, fontsize=fontsize)

plt.subplots_adjust(hspace=0.35,wspace=0.1)
plt.tight_layout()
plt.rc('pdf', fonttype=42)
plt.rcParams.update({'font.size': 15})
plt.savefig('./cdf.pdf')
plt.show()
