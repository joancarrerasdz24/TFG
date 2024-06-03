import WholeBrain.Utils.p_values as pv
import matplotlib.pyplot as plt
import fitting as fit
from setup import subjects
import os

# Cargue sus datos en variables.
# Ejemplo:
'''heterogeneous = [1., 2., 3., 4., 5.]
homogeneous = [6., 7., 8., 9., 10.]
shuffled = [11., 12., 13., 14., 15.]'''

#heterogeneous = fit.heterogeneous_results
#homogeneous = fit.homogeneous_results
#shuffled = fit.shuffled_results


def showPolotResults(burden, heterogeneous_hc, homogeneous_hc, shuffled_hc, heterogeneous_mci, homogeneous_mci, shuffled_mci, heterogeneous_ad, homogeneous_ad, shuffled_ad):
    # Etiquetas para los ejes
    labels = ['Heterogeneous', 'Homogeneous', 'Shuffled']

    # Función para generar el boxplot con p-valores

    data_hc = {'Heterogeneous': heterogeneous_hc, 'Homogeneous': homogeneous_hc, 'Shuffled': shuffled_hc}
    data_mci = {'Heterogeneous': heterogeneous_mci, 'Homogeneous': homogeneous_mci, 'Shuffled': shuffled_mci}
    data_ad = {'Heterogeneous': heterogeneous_ad, 'Homogeneous': homogeneous_ad, 'Shuffled': shuffled_ad}
    # data = {'HC': hc, 'MCI': mci, 'AD': ad}
    print("HC Results")
    pv.plotComparisonAcrossLabels2(data_hc, graphLabel='Comparation of HC data for ' + burden)
    print("MCI Results")
    pv.plotComparisonAcrossLabels2(data_mci, graphLabel='Comparation of MCI data for ' + burden)
    print("AD Results")
    pv.plotComparisonAcrossLabels2(data_ad, graphLabel='Comparation of AD data for ' + burden)

burdens = ["Tau", "ABeta"]

for burden in burdens:
    if burden == "Tau":
        #HC
        heterogeneous_hc = [0.3000732600732601, 0.24279434850863424, 0.2403976975405547, 0.22273155416012558, 0.2966823652537938, 0.18313971742543167, 0.24139194139194142, 0.39342752485609633, 0.14582940868655156, 0.22219780219780222, 0.29493458922030347, 0.3157718472004186, 0.2542647828362114, 0.10648874934589214, 0.1006384092098378, 0.21937205651491365, 0.2381266352694924]
        homogeneous_hc = [0.1696389324960753, 0.22748299319727894, 0.1467817896389325, 0.13920460491889064, 0.18667713239141814, 0.17521716378859237, 0.27439037153322876, 0.27383568812140235, 0.1963474620617478, 0.23783359497645212, 0.41354264782836203, 0.3374568288854003, 0.22641548927263214, 0.06515960230245944, 0.20950287807430665, 0.310559916274202, 0.22421768707482992]
        shuffled_hc = [0.19341705913134483, 0.14496075353218213, 0.30282574568288856, 0.2469178440607012, 0.3192150706436421, 0.26508634222919936, 0.26160125588697014, 0.24276295133437992, 0.21785452642595504, 0.2604186289900576, 0.36804814233385663, 0.2707064364207221, 0.2037990580847724, 0.2946729461015175, 0.12336996336996338, 0.1859654631083203, 0.18012558869701725]

        #MCI
        heterogeneous_mci = [0.14787022501308217, 0.3217687074829932, 0.10246991104133962, 0.1791732077446363, 0.1868236525379382, 0.13569858712715857, 0.17354264782836212, 0.24318158032443749, 0.0741287284144427]
        homogeneous_mci = [0.16774463631606495, 0.16406070120355837, 0.09320774463631605, 0.2582417582417582, 0.13551020408163267, 0.142145473574045, 0.16163265306122448, 0.1716902145473574, 0.0767242281527996]
        shuffled_mci = [0.18035583464154892, 0.17663003663003662, 0.15629513343799056, 0.26631083202511774, 0.22315018315018315, 0.14742019884877028, 0.09672422815279959, 0.2789010989010989, 0.1581894296180011]

        #AD
        heterogeneous_ad = [0.2674725274725275, 0.17179487179487182, 0.19978021978021976, 0.29025641025641025, 0.18637362637362637, 0.19403453689167977, 0.14933542647828368, 0.2040711669283098, 0.18768184196755627, 0.09008895866038724]
        homogeneous_ad = [0.2028990057561486, 0.26551543694400837, 0.3399581371009942, 0.18555729984301417, 0.12410256410256404, 0.22869701726844585, 0.26769230769230773, 0.23731030873888015, 0.1947148090005233, 0.07254840397697541]
        shuffled_ad = [0.30394557823129253, 0.15771847200418632, 0.27432757718472, 0.240690737833595, 0.1525484039769754, 0.14945054945054947, 0.14848770277341705, 0.27060177917320777, 0.24905285190999482, 0.1282051282051282]


        showPolotResults(burden, heterogeneous_hc, homogeneous_hc, shuffled_hc, heterogeneous_mci, homogeneous_mci, shuffled_mci, heterogeneous_ad, homogeneous_ad, shuffled_ad)



    elif burden == "ABeta":
        #HC
        heterogeneous_hc = [0.1318785975928833, 0.20699110413396127, 0.31593929879644167, 0.15377289377289372, 0.4320983778126635, 0.3138880167451596, 0.3333228676085819, 0.3468027210884354, 0.1581998953427525, 0.27332286760858193, 0.24190476190476196, 0.19384615384615383, 0.22727367870225015, 0.09443223443223447, 0.1157613814756672, 0.33267399267399267, 0.18983778126635276]
        homogeneous_hc = [0.14589220303506018, 0.19387755102040816, 0.21089481946624805, 0.14195709052851901, 0.34575614861329146, 0.22691784406070126, 0.27165881737310305, 0.49778126635269493, 0.28035583464154895, 0.27544740973312404, 0.34857142857142853, 0.22579801151229725, 0.2400209314495029, 0.10565149136577712, 0.12738880167451594, 0.19498691784406064, 0.20545264259549967]
        shuffled_hc = [0.18327577184720045, 0.26016745159602306, 0.3235688121402407, 0.15574045002616432, 0.2560858189429618, 0.1695447409733124, 0.2091470434327577, 0.2810675039246468, 0.24985871271585558, 0.2453898482469911, 0.33333333333333337, 0.1968602825745683, 0.2722344322344322, 0.1597069597069597, 0.12007326007326008, 0.28028257456828887, 0.21565672422815274]

        #MCI
        heterogeneous_mci = [0.13075876504447936, 0.21619047619047616, 0.17997906855049714, 0.18406070120355836, 0.2631501831501831, 0.1439141810570382, 0.16881214024071167, 0.21059131344845627, 0.13215070643642068]
        homogeneous_mci = [0.14294086865515432, 0.2610256410256411, 0.10970172684458397, 0.24814233385661955, 0.15551020408163263, 0.15042386185243328, 0.13567765567765566, 0.1542229199372056, 0.19669283097854523]
        shuffled_mci = [0.30982731554160126, 0.19046572475143897, 0.14298273155416014, 0.1839246467817896, 0.2590162218733647, 0.16450026164311882, 0.13116692830978544, 0.1846467817896389, 0.1738147566718995]

        #AD
        heterogeneous_ad = [0.15059131344845633, 0.14616431187859757, 0.2898587127158555, 0.20493982208267925, 0.1003139717425432, 0.1654945054945055, 0.2219466248037677, 0.208487702773417, 0.214850863422292, 0.06858189429618002]
        homogeneous_ad = [0.2192569335426478, 0.27069597069597073, 0.21362637362637366, 0.2765149136577708, 0.11709052851909996, 0.10088958660387226, 0.1962323390894819, 0.2664887493458922, 0.26375719518576657, 0.12847723704866562]
        shuffled_ad = [0.2476085818942962, 0.3264887493458922, 0.22509680795395082, 0.17965463108320257, 0.13658817373103085, 0.20589220303506017, 0.22857142857142854, 0.19115646258503405, 0.31042386185243326, 0.13298796441653585]

        showPolotResults(burden, heterogeneous_hc, homogeneous_hc, shuffled_hc, heterogeneous_mci, homogeneous_mci, shuffled_mci, heterogeneous_ad, homogeneous_ad, shuffled_ad)
