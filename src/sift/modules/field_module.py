
import cv2 

class FieldParser:
    def __init__(self):
        pass 

    def parse(self, ocr_output, field_infos, iou_threshold=0.7):
        """parse field infor from template

        Args:
            ocr_output (list[dict]): 
                [
                    {
                        'box': [xmin, ymin, xmax, ymax]
                        'text': str
                    }
                ]
                
                
            field_infos (list[dict]): _description_
                - example:
                [
                    {
                        'id' : 'field_1'
                        'box': [xmin, ymin, xmax, ymax],
                    }
                [

        Returns:
            field text:
                [
                    {
                        'id' : 'field_1'
                        'box': [xmin, ymin, xmax, ymax],
                        'text': 'abc'
                    }
                [
        """
        for field_item in field_infos:
            if 'list_words' not in field_item:
                field_item['list_words'] = []

        for ocr_item in ocr_output:
            box = ocr_item['box']
            for field_item in field_infos:

                field_name = field_item['id']
                field_box = field_item['box']
                iou = self.cal_iou_custom(box, field_box)
                # if iou > 0:
                #     print(iou, ocr_item)
                if iou > iou_threshold:
                    
                    field_item['list_words'].append(ocr_item)
                    break # break if find field box

        
        for field_item in field_infos:
            list_words = field_item['list_words']
            list_words = sorted(list_words, key=lambda item: item['box'][0])
            field_text = " ".join([item['text'] for item in list_words])
            field_item['text'] = field_text
        

        return field_infos




    def cal_iou_custom(self, box_A, box_B):
        """ calculate iou between two boxes
        union = smaller box between two boxes

        Args:
            box_A (list): _description_
            box_B (list): _description_

        Returns:
            (float): iou value 
        """
        
        area1 = (box_A[2] - box_A[0]) * (
            box_A[3] - box_A[1])
        area2 = (box_B[2] - box_B[0]) * (
            box_B[3] - box_B[1])

        xmin_intersect = max(box_A[0], box_B[0])
        ymin_intersect = max(box_A[1], box_B[1])
        xmax_intersect = min(box_A[2], box_B[2])
        ymax_intersect = min(box_A[3], box_B[3])
        if xmax_intersect < xmin_intersect or ymax_intersect < ymin_intersect:
            area_intersect = 0
        else:
            area_intersect = (xmax_intersect - xmin_intersect) * (
                ymax_intersect - ymin_intersect
            )


        # union = area1 + area2 - area_intersect
        union = min(area1, area2)
        if union == 0:
            return 0
        
        iou = area_intersect / union
        
        return iou 


def format_field_info(data, id):
    """{'name': 'field',
              'type': 'text',
              'position': {'top': 1951, 'left': 1173},
              'size': {'width': 1224, 'height': 110}}

    Args:
        data (_type_): _description_
    """

    output = {}
    output['id'] = data['name'] + "_" + str(id)
    xmin, ymin, w, h = data['position']['left'], data['position']['top'], data['size']['width'], data['size']['height']
    output['box'] = [xmin, ymin, xmin+w, ymin+h]
    return output 

def vis_field(img, field_infos):
    for field_item in field_infos:
        box = field_item['box']
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 255,0), thickness=1)

    return img 

def merge_field(image, field_infos):
    pass



if __name__ == "__main__":


    ocr_output = ([(1972, 101, 2324, 231), (1980, 238, 2213, 283), (607, 277, 702, 317), (525, 279, 600, 317), (242, 280, 345, 323), (400, 280, 518, 317), (819, 281, 887, 324), (348, 282, 395, 324), (895, 282, 988, 318), (997, 282, 1076, 325), (712, 282, 812, 317), (1083, 283, 1174, 318), (241, 325, 327, 366), (331, 328, 390, 367), (397, 330, 684, 370), (245, 388, 441, 457), (778, 390, 945, 458), (616, 390, 762, 457), (965, 395, 1176, 459), (458, 395, 599, 457), (1198, 396, 1422, 472), (1574, 397, 1677, 459), (1697, 398, 1883, 460), (1440, 398, 1557, 458), (1584, 518, 2283, 608), (976, 527, 1056, 593), (1046, 530, 1093, 591), (687, 531, 742, 590), (826, 532, 881, 596), (904, 533, 956, 594), (751, 534, 805, 597), (1638, 537, 1687, 597), (1563, 537, 1621, 597), (508, 550, 600, 589), (606, 551, 665, 592), (1295, 551, 1352, 594), (327, 551, 423, 597), (429, 552, 502, 590), (240, 552, 322, 597), (1354, 554, 1436, 597), (1442, 555, 1549, 597), (1040, 619, 1228, 709), (1504, 626, 1631, 707), (617, 635, 710, 675), (719, 636, 873, 681), (239, 636, 299, 682), (537, 636, 611, 675), (1276, 637, 1440, 690), (371, 637, 444, 674), (302, 638, 365, 675), (450, 644, 532, 675), (1188, 703, 1374, 778), (1631, 704, 1743, 781), (1423, 707, 1566, 777), (675, 718, 768, 759), (777, 720, 917, 764), (238, 720, 299, 766), (595, 720, 669, 759), (492, 721, 589, 764), (371, 721, 486, 764), (301, 721, 365, 759), (925, 722, 1038, 760), (1663, 789, 1718, 841), (1488, 791, 1583, 835), (1256, 792, 1347, 840), (562, 792, 656, 835), (1587, 793, 1661, 835), (483, 794, 558, 835), (737, 794, 809, 834), (238, 795, 313, 840), (814, 795, 897, 834), (1721, 795, 1806, 835), (659, 795, 734, 842), (1181, 796, 1254, 836), (317, 796, 391, 835), (1350, 796, 1383, 837), (1812, 796, 1895, 841), (1012, 797, 1074, 841), (905, 797, 1007, 834), (1079, 797, 1175, 841), (1386, 797, 1482, 839), (396, 802, 477, 833), (239, 868, 300, 926), (336, 876, 418, 922), (302, 876, 332, 917), (424, 877, 521, 922), (822, 877, 897, 922), (657, 877, 726, 918), (730, 877, 817, 919), (527, 878, 650, 922), (681, 957, 741, 1007), (1806, 959, 1869, 1007), (1355, 960, 1425, 1006), (1695, 960, 1732, 1002), (1427, 960, 1488, 1001), (506, 960, 568, 1007), (752, 960, 821, 1007), (824, 960, 887, 1002), (300, 961, 367, 1007), (370, 961, 430, 1001), (433, 961, 503, 1001), (1734, 961, 1803, 1001), (1631, 962, 1692, 1002), (1029, 963, 1090, 1002), (1493, 963, 1626, 1006), (892, 964, 1023, 1006), (298, 1042, 356, 1086), (359, 1045, 439, 1089), (443, 1046, 510, 1085), (515, 1046, 647, 1090), (303, 1128, 519, 1176), (1360, 1129, 1610, 1176), (519, 1210, 607, 1260), (1358, 1211, 1463, 1253), (300, 1212, 393, 1255), (393, 1213, 515, 1254), (1467, 1214, 1540, 1259), (1992, 1260, 2326, 1347), (934, 1269, 1415, 1353), (990, 1280, 1037, 1345), (912, 1284, 961, 1350), (243, 1285, 303, 1341), (1415, 1291, 1481, 1341), (828, 1291, 873, 1347), (694, 1292, 738, 1343), (1484, 1293, 1594, 1341), (759, 1294, 799, 1345), (540, 1294, 662, 1344), (394, 1294, 535, 1341), (303, 1295, 389, 1342), (242, 1371, 301, 1423), (303, 1379, 416, 1422), (239, 1474, 302, 1533), (302, 1478, 348, 1527), (670, 1480, 742, 1524), (439, 1480, 538, 1529), (745, 1480, 851, 1523), (349, 1481, 434, 1530), (857, 1481, 959, 1524), (541, 1483, 666, 1528), (1561, 1558, 1618, 1606), (1248, 1558, 1305, 1609), (946, 1559, 1005, 1608), (527, 1559, 586, 1608), (835, 1561, 933, 1607), (1305, 1562, 1367, 1612), (1006, 1563, 1124, 1607), (589, 1563, 664, 1607), (757, 1563, 832, 1607), (240, 1565, 330, 1608), (1368, 1565, 1434, 1606), (442, 1566, 517, 1608), (1129, 1566, 1234, 1606), (1440, 1566, 1556, 1604), (335, 1567, 438, 1607), (1621, 1570, 1723, 1611), (669, 1573, 752, 1606), (1855, 1642, 1909, 1693), (1246, 1642, 1309, 1694), (1913, 1645, 1992, 1688), (302, 1647, 362, 1698), (1996, 1648, 2084, 1688), (1416, 1648, 1506, 1689), (1311, 1648, 1412, 1695), (1581, 1651, 1608, 1690), (364, 1652, 441, 1692), (1667, 1652, 1700, 1689), (304, 1733, 407, 1775), (410, 1733, 499, 1781), (244, 1811, 300, 1863), (302, 1813, 359, 1861), (359, 1815, 441, 1866), (442, 1817, 490, 1859), (559, 1818, 658, 1859), (491, 1818, 557, 1864), (979, 1896, 1065, 1948), (904, 1897, 977, 1942), (408, 1900, 493, 1948), (300, 1901, 405, 1951), (647, 1905, 676, 1942), (556, 1905, 584, 1942), (518, 1909, 542, 1939), (598, 1909, 627, 1940), (1786, 1941, 1889, 2034), (1922, 1949, 2024, 2054), (2043, 1951, 2191, 2041), (1450, 1952, 1599, 2038), (1298, 1957, 1420, 2046), (1641, 1968, 1765, 2036), (243, 1973, 305, 2033), (1078, 1978, 1169, 2029), (794, 1980, 840, 2025), (715, 1981, 793, 2029), (941, 1982, 1023, 2030), (305, 1982, 409, 2033), (842, 1983, 938, 2031), (413, 1984, 659, 2032), (1024, 1986, 1077, 2030), (660, 1988, 713, 2031), (306, 2055, 344, 2097), (239, 2057, 309, 2097), (2274, 2083, 2328, 2118), (2208, 2083, 2272, 2119), (1980, 2083, 2053, 2123), (2125, 2084, 2203, 2118), (1326, 2085, 1376, 2126), (2055, 2085, 2123, 2118), (1911, 2085, 1979, 2124), (1249, 2085, 1325, 2121), (1639, 2086, 1739, 2124), (1479, 2086, 1551, 2120), (1170, 2086, 1247, 2126), (1824, 2086, 1908, 2119), (1025, 2086, 1091, 2122), (354, 2087, 428, 2127), (1379, 2087, 1478, 2120), (783, 2087, 861, 2122), (1743, 2087, 1821, 2119), (1555, 2087, 1633, 2119), (1093, 2087, 1168, 2121), (945, 2087, 1022, 2122), (863, 2087, 942, 2122), (593, 2087, 674, 2126), (677, 2088, 780, 2125), (520, 2089, 591, 2124), (430, 2089, 466, 2124), (467, 2089, 518, 2126), (816, 2127, 853, 2166), (639, 2128, 694, 2167), (945, 2128, 1050, 2165), (695, 2129, 734, 2166), (351, 2129, 416, 2165), (735, 2129, 815, 2165), (597, 2130, 639, 2164), (518, 2130, 596, 2163), (418, 2130, 516, 2166), (853, 2131, 942, 2165), (335, 2142, 357, 2170), (1768, 2163, 1820, 2198), (2230, 2163, 2268, 2200), (2270, 2165, 2329, 2199), (1126, 2165, 1194, 2205), (1659, 2165, 1728, 2203), (2133, 2165, 2227, 2202), (1823, 2165, 1890, 2202), (2092, 2165, 2130, 2198), (1984, 2165, 2027, 2199), (1536, 2166, 1598, 2199), (1411, 2166, 1465, 2205), (1893, 2166, 1981, 2202), (1730, 2166, 1766, 2199), (2030, 2167, 2089, 2197), (1196, 2167, 1288, 2201), (1342, 2167, 1410, 2205), (1045, 2167, 1123, 2204), (991, 2167, 1044, 2206), (882, 2168, 953, 2206), (1467, 2168, 1533, 2199), (355, 2168, 410, 2204), (679, 2169, 768, 2206), (531, 2169, 616, 2203), (1291, 2169, 1340, 2201), (770, 2169, 816, 2203), (816, 2169, 881, 2203), (953, 2169, 990, 2204), (617, 2169, 678, 2204), (461, 2169, 529, 2204), (1601, 2170, 1657, 2199), (411, 2170, 459, 2204), (333, 2172, 353, 2202), (1657, 2206, 1705, 2243), (2000, 2206, 2071, 2239), (1772, 2206, 1837, 2239), (1913, 2206, 1997, 2243), (1252, 2206, 1305, 2244), (1841, 2207, 1908, 2241), (1560, 2207, 1654, 2239), (1427, 2207, 1486, 2240), (1489, 2207, 1557, 2240), (807, 2207, 854, 2246), (1068, 2207, 1153, 2241), (1156, 2207, 1250, 2243), (987, 2208, 1066, 2245), (1707, 2208, 1768, 2242), (596, 2208, 668, 2247), (668, 2208, 720, 2243), (722, 2208, 807, 2242), (926, 2208, 986, 2242), (855, 2208, 925, 2246), (1309, 2208, 1424, 2243), (501, 2209, 594, 2247), (436, 2210, 499, 2244), (353, 2216, 434, 2248), (1462, 2245, 1521, 2285), (1734, 2245, 1812, 2284), (1213, 2246, 1288, 2281), (1092, 2246, 1148, 2287), (1335, 2246, 1388, 2280), (1816, 2247, 1919, 2283), (897, 2247, 977, 2287), (1148, 2247, 1210, 2282), (1618, 2247, 1680, 2285), (737, 2247, 806, 2283), (1391, 2247, 1459, 2285), (1683, 2247, 1731, 2280), (979, 2248, 1090, 2287), (675, 2248, 735, 2282), (809, 2248, 894, 2282), (1923, 2248, 1991, 2279), (1525, 2248, 1614, 2284), (1291, 2249, 1332, 2281), (613, 2249, 674, 2288), (545, 2250, 611, 2288), (472, 2250, 542, 2288), (417, 2250, 470, 2284), (356, 2250, 415, 2284), (335, 2252, 352, 2284), (697, 2326, 793, 2369), (259, 2326, 356, 2378), (635, 2326, 694, 2374), (467, 2327, 538, 2368), (543, 2329, 631, 2368), (365, 2332, 462, 2375), (947, 2401, 1008, 2456), (1247, 2404, 1306, 2455), (1564, 2406, 1619, 2449), (839, 2407, 934, 2451), (1307, 2408, 1368, 2457), (1010, 2409, 1125, 2452), (535, 2409, 587, 2463), (1131, 2410, 1236, 2451), (761, 2411, 834, 2451), (1443, 2411, 1558, 2448), (1372, 2412, 1437, 2449), (594, 2413, 666, 2452), (244, 2413, 331, 2453), (1623, 2415, 1725, 2455), (339, 2416, 439, 2452), (447, 2416, 518, 2453), (673, 2420, 754, 2451), (1930, 2518, 1980, 2565), (1879, 2519, 1926, 2565), (1639, 2521, 1720, 2558), (1726, 2521, 1775, 2565), (1780, 2522, 1873, 2563), (759, 2524, 808, 2569), (812, 2525, 862, 2563), (672, 2525, 754, 2562), (690, 2613, 798, 2723), (1626, 2632, 1837, 2756), (465, 2727, 650, 2873), (878, 2734, 997, 2826), (695, 2756, 839, 2841), (1868, 2760, 1997, 2853), (1686, 2785, 1833, 2862), (1488, 2787, 1667, 2942), (2031, 3036, 2080, 3080), (1819, 3037, 1870, 3082), (2212, 3040, 2305, 3081), (1683, 3040, 1797, 3084), (1879, 3040, 1979, 3084), (1275, 3042, 1337, 3082), (2310, 3042, 2336, 3082), (2089, 3042, 2204, 3081), (1056, 3042, 1137, 3082), (1982, 3042, 2011, 3084), (1394, 3043, 1481, 3080), (1608, 3043, 1675, 3080), (1489, 3044, 1601, 3080), (1341, 3045, 1389, 3082), (1000, 3045, 1050, 3083), (1191, 3046, 1270, 3087), (1142, 3046, 1186, 3083), (903, 3048, 994, 3088), (739, 3048, 822, 3089), (246, 3048, 322, 3084), (534, 3049, 592, 3083), (829, 3049, 896, 3084), (331, 3049, 408, 3089), (417, 3050, 526, 3083), (675, 3050, 733, 3083), (598, 3051, 668, 3089), (346, 3097, 412, 3131), (247, 3100, 335, 3131), (1685, 3136, 1776, 3175), (1276, 3138, 1494, 3182), (1782, 3139, 1841, 3180), (1606, 3140, 1678, 3176), (1502, 3141, 1599, 3181), (1197, 3141, 1268, 3178), (544, 3142, 625, 3178), (1030, 3143, 1102, 3178), (793, 3144, 845, 3179), (959, 3144, 1022, 3183), (852, 3144, 953, 3178), (716, 3144, 788, 3184), (491, 3146, 538, 3185), (632, 3146, 709, 3178), (414, 3147, 484, 3179), (283, 3148, 404, 3186), (1109, 3149, 1189, 3178), (245, 3150, 273, 3180), (2117, 3183, 2206, 3220), (1666, 3185, 1827, 3224), (1961, 3187, 2028, 3221), (2038, 3187, 2106, 3220), (1482, 3188, 1578, 3230), (2216, 3188, 2283, 3220), (1586, 3189, 1658, 3225), (1885, 3189, 1953, 3228), (1222, 3189, 1283, 3233), (1396, 3190, 1476, 3231), (1097, 3190, 1163, 3227), (1168, 3190, 1218, 3228), (1034, 3192, 1093, 3228), (1288, 3192, 1389, 3231), (484, 3192, 547, 3227), (909, 3193, 1027, 3233), (829, 3193, 903, 3228), (638, 3194, 718, 3233), (554, 3195, 630, 3227), (1832, 3195, 1880, 3224), (726, 3195, 820, 3233), (420, 3195, 477, 3228), (290, 3196, 411, 3234), (243, 3197, 281, 3230), (1376, 3236, 1426, 3274), (1299, 3238, 1371, 3279), (1730, 3239, 1808, 3278), (1671, 3239, 1724, 3273), (1555, 3239, 1664, 3278), (1432, 3239, 1547, 3278), (1200, 3240, 1292, 3280), (971, 3241, 1083, 3280), (1091, 3241, 1192, 3275), (865, 3241, 962, 3275), (779, 3242, 858, 3280), (656, 3243, 707, 3275), (604, 3243, 650, 3275), (522, 3243, 598, 3281), (714, 3243, 771, 3275), (329, 3244, 362, 3276), (370, 3245, 462, 3283), (246, 3246, 320, 3276), (470, 3250, 517, 3281), (2185, 3354, 2332, 3379), (876, 3357, 948, 3387), (734, 3358, 807, 3387), (811, 3358, 871, 3391), (522, 3358, 592, 3388), (679, 3359, 729, 3391), (599, 3359, 674, 3387), (461, 3361, 517, 3388), (250, 3362, 326, 3393), (367, 3362, 455, 3388), (331, 3363, 362, 3393), (2290, 3391, 2332, 3416), (380, 3398, 626, 3427), (249, 3400, 374, 3427)], ['FWD', 'insurance', 'hiểm', 'Bảo', 'Công', 'TNHH', 'thọ', 'ty', 'FWD', 'Việt', 'Nhân', 'Nam', 'Mẫu', 'số:', 'POS01_2022.09', 'Phiếu', 'Điều', 'Cầu', 'Chỉnh', 'Yêu', 'Thông', 'Cá', 'Nhân', 'Tin', '0357788028', '3', '0', '13', '2', '3', '4', '13', '10', 'hiểm', 'số:', 'Số', 'đồng', 'bảo', 'Hợp', 'điện', 'thoại:', 'Nguyễn', 'Hiệp', 'hiểm', '(BMBH):', 'Họ', 'bảo', 'hoan', 'Bên', 'tên', 'mua', 'Nguyễn', 'Hiệp', 'Hoàng', 'hiểm', '(NĐBH)', 'Họ', 'bảo', 'được', 'Người', 'tên', 'chính:', '(x)', 'đánh', '(cắc)', 'hiểm', 'dấu', 'bảo', 'cầu', 'Tôi,', 'điều', 'dưới', 'yêu', 'của', 'Bên', 'ô', 'đây:', 'nội', 'chỉnh', 'dung', 'được', 'mua', 'X', 'Cập', 'L', 'Nhật', 'Lạc', 'Tin', 'Liên', 'Thông', '0', 'lạc', 'Địa', '&', 'chỉ', 'lạc', 'Địa', 'chỉ', 'Địa', 'chỉ', 'liên', 'liên', 'trú', 'trú', 'thường', 'thường', 'Số', 'nhà,', 'tên', 'đường:', 'Phường/Xã:', 'Quận/Huyện:', 'phố:', 'Quốc', 'Tỉnh/', 'Thành', 'gia:', 'T', '345968', '3', '7', '8', '(cố', '2', 'o', 'định):', '3', 'động):', 'thoại(di', 'Điện', '0', 'Email:', 'X', 'II.', 'Tin', 'Nhật', 'Nhân', 'Cập', 'Thân', 'Thông', 'bổ', '0', '0', '0', 'hiểm', 'Họ', 'NĐBH', 'Bên', 'bảo', 'Điều', 'tên', 'cho', 'chính', 'NĐBH', 'chỉnh', 'sung:', 'mua', '0', '0', 'Giới', 'Họ', 'tính:', 'sinh:', 'Ngày', '/', 'tên:', 'J', 'Quốc', 'tịch:', '0', 'Số', 'giấy', 'tờ', 'thân:', 'tùy', 'cấp:', 'Nơi', 'cấp:', 'Ngày', '/', '/', '✪', '_.', 'thể', 'điện', 'thoại', 'doan', 'Kinh', 'Sum', '_', 'thể):', 'tả', '(mô', 'Việc', 'Nghề', 'công', 'nghiệp/Chức', 'cụ', 'vụ', 'ý:', 'Lưu', 'thẻ', 'Các', 'Giấy', 'sinh/', 'Hộ', 'khai', 'đội/', 'dân/', 'Chứng', 'Khai', 'công', 'Quân', 'Căn', 'Giấy', 'chiếu/', 'minh', 'minh', 'sinh/', 'cước', 'dân/', 'nhân', 'gồm:', 'Chứng', 'thân', 'tờ', 'tùy', 'lý', 'giá', 'đương', 'trị', 'ban', 'pháp', 'CÓ', 'khác', 'ngành', 'tương', '✪', 'thể', 'từ', 'liên', 'Quý', 'giấy', 'chứng', 'hiện', 'và', 'tin', 'bản', 'gửi', 'thông', 'tờ', 'mới', 'khách', 'lòng', 'thân,', 'tùy', 'giấy', 'kèm', 'Đối', 'thông', 'chỉnh', 'vui', 'tin', 'trên', 'tờ', 'các', 'điều', 'sao', 'với', '-', 'Họ', 'sinh.', 'Giới', 'Ngày', 'địa', 'tính,', 'chỉnh:', 'nếu', 'điều', 'hộ', 'chính', 'quyền', 'nhận', 'tên,', 'định', 'cải', 'chính', 'xác', 'tịch;', 'phương', 'Quyết', 'như', 'quan', 'đổi,', 'nghề', 'hiểm', 'phí', 'thể', 'nghiệp', 'nghề', 'bảo', 'ứng', 'điểu', 'thay', 'với', 'nghiệp,', 'cầu', 'chỉnh', 'mới.', 'tương', 'CÓ', 'yêu', 'hiện', 'thực', 'khi', 'Sau', '.', 'Mẫu', 'XII.', 'Ký', 'Đổi', 'Chữ', 'Thay', 'X', '0', 'bổ', 'hiểm', 'Họ', 'NĐBH', 'X', 'chính', 'bảo', 'NĐBH', 'tên', 'Bên', 'Điều', 'sung:', 'chỉnh', 'cho', 'mua', 'lại', 'ký', 'Chữ', 'ký', 'đăng', 'ký', 'cũ', 'Chữ', 'Hiệp', 'Huệp', 'Nguyễn', 'Hiệp', 'Hoàng', 'Hiệp', 'Hoàng', 'Nguyễn', '0', '0', 'đồng', 'phẩm:', 'Đồng', 'tắc', 'Ý', 'Không', 'hiểu', 'Ý', 'Điều', 'sản', 'khoản', 'và', 'đã', 'Quy', 'rõ', 'nhận', 'lòng', 'Nếu', 'lăn', 'xác', 'Quý', 'khách', 'vui', 'tay,', 'Kết', 'Cam', 'hiểm', 'hiểm/Người', 'ký.', 'bảo', 'được', 'bảo', 'mẫu', 'Bên', 'do', 'tôi,', 'chính', 'đây', 'ký', 'trên', 'chữ', 'Những', 'mua', '1.', 'biểm', 'hiểm/Hồ', 'cầu', 'bảo', 'đồng', 'nêu', 'bảo', 'yêu', 'ghi', 'Hợp', 'tiết', 'đã', 'chi', 'trong', 'tiết', 'những', 'như', 'đây,', 'trên', 'SƠ', 'cũng', 'chi', 'Những', '2.', 'về', 'luật', 'này.', 'tin', 'thông', 'những', 'pháp', 'nhiệm', 'trước', 'trách', 'chịu', 'tôi', 'và', 'thật', 'xin', 'là', 'đúng', 'trên', 'sự', 'V1.092022', 'Nam', 'FWD', 'Việt', 'hiểm', 'thọ', 'Nhân', 'Bảo', 'Công', 'TNHH', 'ty', '1/2', 'www.fwd.com.vn', 'Website:'])
   
    ocr_output = [
        {
            'box': box,
            'text': text
        }
        for (box, text) in zip(ocr_output[0], ocr_output[1])
    ]

    
    field_infos = [{'id': 'field_0', 'box': [1173, 1951, 2397, 2061]}, {'id': 'field_1', 'box': [457, 1607, 1244, 1726]}, {'id': 'field_2', 'box': [1621, 1092, 2369, 1202]}, {'id': 'field_3', 'box': [1506, 1620, 1864, 1699]}, {'id': 'field_4', 'box': [1062, 1875, 1789, 1959]}, {'id': 'field_5', 'box': [487, 1872, 874, 1957]}, {'id': 'field_6', 'box': [665, 1781, 1551, 1878]}, {'id': 'field_7', 'box': [2085, 1625, 2386, 1711]}, {'id': 'field_8', 'box': [608, 1192, 1360, 1264]}, {'id': 'field_9', 'box': [415, 1345, 2337, 1465]}, {'id': 'field_10', 'box': [501, 1712, 1250, 1791]}, {'id': 'field_11', 'box': [1725, 1546, 2428, 1633]}, {'id': 'field_12', 'box': [1599, 1261, 2330, 1349]}, {'id': 'field_13', 'box': [667, 1263, 1402, 1357]}, {'id': 'field_14', 'box': [1549, 1189, 2334, 1268]}, {'id': 'field_15', 'box': [524, 1103, 1359, 1204]}, {'id': 'field_16', 'box': [657, 1006, 2477, 1117]}, {'id': 'field_17', 'box': [876, 603, 2332, 717]}, {'id': 'field_18', 'box': [1041, 691, 2340, 801]}, {'id': 'field_19', 'box': [1567, 512, 2296, 602]}, {'id': 'field_20', 'box': [673, 504, 1271, 609]}]
    
    # field_infos = [{'name': 'field',
    #           'type': 'text',
    #           'position': {'top': 1951, 'left': 1173},
    #           'size': {'width': 1224, 'height': 110}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1607, 'left': 457},
    #              'size': {'width': 787, 'height': 119}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1092, 'left': 1621},
    #              'size': {'width': 748, 'height': 110}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1620, 'left': 1506},
    #              'size': {'width': 358, 'height': 79}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1875, 'left': 1062},
    #              'size': {'width': 727, 'height': 84}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1872, 'left': 487},
    #              'size': {'width': 387, 'height': 85}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1781, 'left': 665},
    #              'size': {'width': 886, 'height': 97}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1625, 'left': 2085},
    #              'size': {'width': 301, 'height': 86}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1192, 'left': 608},
    #              'size': {'width': 752, 'height': 72}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1345, 'left': 415},
    #              'size': {'width': 1922, 'height': 120}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1712, 'left': 501},
    #              'size': {'width': 749, 'height': 79}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1546, 'left': 1725},
    #              'size': {'width': 703, 'height': 87}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1261, 'left': 1599},
    #              'size': {'width': 731, 'height': 88}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1263, 'left': 667},
    #              'size': {'width': 735, 'height': 94}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1189, 'left': 1549},
    #              'size': {'width': 785, 'height': 79}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1103, 'left': 524},
    #              'size': {'width': 835, 'height': 101}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 1006, 'left': 657},
    #              'size': {'width': 1820, 'height': 111}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 603, 'left': 876},
    #              'size': {'width': 1456, 'height': 114}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 691, 'left': 1041},
    #              'size': {'width': 1299, 'height': 110}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 512, 'left': 1567},
    #              'size': {'width': 729, 'height': 90}},
    #             {'name': 'field',
    #              'type': 'text',
    #              'position': {'top': 504, 'left': 673},
    #              'size': {'width': 598, 'height': 105}}]

    # field_infos = [
    #     format_field_info(field_item, idx)
    #     for idx, field_item in enumerate(field_infos)
    # ]

    print(field_infos)

    img = cv2.imread("/mnt/ssd500/hoanglv/Projects/FWD/template_matching_hoanglv/assest/form_1_edit_personal_info/Scan47_0.jpg")
    img = vis_field(img, field_infos)
    cv2.imwrite("vis_field.jpg", img)
    print(ocr_output[0])
    print(field_infos[0])
    parser = FieldParser()
    field_outputs = parser.parse(ocr_output, field_infos)

    for field_item in field_outputs:
        if len(field_item['list_words']) > 0:
            print(field_item['id'], field_item['text'])