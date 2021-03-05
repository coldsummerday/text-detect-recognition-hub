from texthub.modules.rec_heads.srn_head import *
import torch
conv_features = torch.rand((16,256,512))
position_embedding_layer = PositionalEncoding(512, n_position=256)

trans_unit = TransformerUnit(dim_model=512,dim_inner_hid=512,num_heads=8,dim_k=512,dim_v=512)
b = position_embedding_layer(conv_features)
c = trans_unit(b)

pvam = PVAM(channels=512)
pvam_features = pvam(c)
gsrm = GSRM()
gsrm_features, word_out, gsrm_out=gsrm(pvam_features)
vsfd=VSFD(dim_hidden=512)
final_features = vsfd(pvam_features,gsrm_features)

from texthub.modules.rec_heads.srn_head import *
import torch
charsets = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~¥§®°±²´·»ÉËÓ×Üàáäèéìíòó÷úüāēīōūǐǒɔɡʌˇˉˋΛΟΦΩαβεθμπЗИЙПЯг—―‖‘’“”•…‧‰′″※℃№™ⅡⅢⅣⅧ←↑→↓⇋∈∑√∞∣∧∩∫∶≈≠≤≥⊙⊥①②③④⑧⑴⑵⑶─│┅┌├█▎▏▕■□▪▲△▼◆◇○◎●◥★☆❋❤、。〃々〆〇〈〉《》「」『』【】〒〓〔〕〖〗〝〞のィエサシジスダデプマユㄧㄱㆍ㎏㎡㐂㐱㙟㴪㸃䖝䝉一丁七丄万丈三上下不与丐丑专且丕世丘丙业丛东丝丞両丢两严丧丨个丫中丰串临丶丸丹为主丼丽举丿乃久么义之乌乍乎乏乐乒乓乔乖乘乙乚乜九乞也习乡书买乱乳乸乾了予争事二亍于亏云互亓五井亘亚些亜亞亟亡亢交亥亦产亨亩享京亭亮亲亳亵人亻亿什仁仂仃仄仅仆仇今介仍从仑仓仔仕他仗付仙仝仞仟仡代令以仨仪们仰仲件价仺任份仿企伈伉伊伍伎伏伐休众优伙会伞伟传伢伤伦伧伪伫伯估伱伴伶伸伺似伽佃但位低住佐佑体何佗佘余佚佛作佝佞佟你佣佤佧佩佬佯佰佳佶佻佼使侃侄來侈侉例侍侏侑侗供依侞侠侣侥侦侧侨侩侬侮侯侵侷便係促俄俊俍俎俏俐俑俗俘俚保俞俟信俨俩俪俫俬俭修俯俱俳俵俸俺俾倆倉個倌倍倏們倒倔倘候倚倜借倡倦倩倪倫倬倭债值倾偃假偈偉偌偎偏偕做停健偶偷偻偽偿傀傅傈傍傘備傢傣傥储傩催傲傳傷傻像僑僚僧僮僳僵價僻儀億儋儒儡儥優儱儿兀允元兄充兆先光克免兑兒兔兖党兜兢入內全兩八公六兮兰共关兴兵其具典兹养兼兽兿冀内冈冉册再冏冒冕冗写冚军农冠冢冤冥冬冮冯冰冱冲决况冶冷冻冼冽净凄凅准凇凈凉凋凌凍减凑凖凛凝几凡凤処凫凬凭凯凰凱凳凶凸凹出击函凿刀刁刂刃分切刈刊刍刑划列刘则刚创初删判刨利刪别刮到制刷券刹刺刻刽刿剁剂剃削剌前剎剐剑剔剖剛剡剥剧剩剪副割創剽剿劃劇劈劉劍劑力劝办功加务劣动助努劫劭励劲劳劵劾势勁勃勅勇勉勋勐勑勒動勘務勝募勢勤勰勺勾勿匀包匆匈匋匍匐匕化北匙匝匠匡匣匪匮匯匹区医匾匿區十千卅升午卉半卍华协卑卒卓協单卖南博卜卞卟占卡卢卤卦卧卩卫卯印危即却卵卷卸卿厂厄厅历厉压厌厍厕厘厚厝原厢厥厦厨厩厮厲去县叁参參叄又叉及友双反发叔取受变叙叛叟叠叢口古句另叨叩只叫召叭叮可台叱史右叵叶号司叹叻叼叽吁吃各吆合吉吊吋同名后吏吐向吒吓吕吖吗君吝吞吟吠吡否吧吨吩含听吭吮启吱吲吴吵吸吹吻吼吾呀呃呆呈告呋呎呐呓呔呕呗员呙呛呜呢呤呦周呱呲味呵呶呷呸呻呼命咀咂咄咅咆咋和咎咏咐咑咒咔咕咖咙咚咛咝咣咤咦咧咨咩咪咫咬咭咯咱咳咴咸咻咽咾咿哀品哄哆哇哈哉哌响哎哐哑哒哓哔哗哚哝哞哟員哥哦哧哨哩哪哭哮哲哺哼哽唁唆唇唉唏唐唑唔唘唛唠唢唤唧唬售唯唰唱唳唷唻唾唿啃啄商啉啊問啐啓啕啖啜啟啡啤啥啦啧啪啫啬啭啮啰啵啶啷啸啻啼啾喀喂喃善喆喇喉喊喋喏喔喘喙喜喝喟喧喫喬單喰喱喳喵喷喹喻喽嗄嗅嗉嗑嗒嗓嗔嗖嗛嗜嗝嗞嗟嗡嗣嗤嗥嗦嗨嗪嗫嗬嗮嗯嗲嗳嗵嗷嗽嘀嘁嘈嘉嘌嘎嘘嘛嘞嘟嘢嘣嘤嘧嘬嘭嘱嘲嘴嘶嘹嘻嘿噁噌噎噔噗噘噙噜噢噤器噩噪噬噱噶噹噻噼嚅嚎嚏嚒嚓嚕嚞嚣嚯嚴嚷嚼囆囊囍囔囗囚四回囟因囡团囤囫园困囱围囵囹固国图囿圃圄圆圈國圍園圖圗團圜土圣圧在圩圭地圳圹场圻圾址坂均坊坋坍坎坏坐坑块坚坛坝坞坟坠坡坤坦坨坩坪坫坭坯坳坶坷坻垂垃垄垅垆型垌垒垓垚垛垟垠垡垢垣垦垩垫垭垮垸埂埃埋埌城埔埕埗埚域埠埭執培基堀堂堃堆堇堑堕堞堡堤堪堰報場堵堷堺塊塌塍塑塔塗塘塞填塬塾境墅墉墊墒墓墙增墟墨墩壁壅壇壐壑壓壕壞壤士壬壮声壳壴壶壹壽夀处夆备夋夌复夏夔夕外夙多夜够夢夥大天太夫夬夭央夯失头夷夸夹夺夾奂奄奇奈奉奋奎奏契奓奔奕奖套奘奚奠奢奥奧奪女奴奶奸她好如妃妄妆妇妈妊妍妒妓妖妗妙妝妞妤妥妨妩妪妫妮妯妲妳妹妻妾姆姊始姌姐姑姒姓委姗姘姚姜姝姣姥姨姬姹姻姿威娃娄娅娆娇娈娉娌娑娓娘娜娟娠娣娥娩娱娲娴娶娼娽婀婆婉婊婕婚婢婦婧婪婭婴婵婶婷婺婿媄媒媚媛媲媳媽媾嫁嫂嫉嫌嫒嫔嫖嫚嫡嫣嫦嫩嬉嬗嬰嬴嬷孀孃子孑孓孔孕孖字存孙孚孛孜孝孟孢季孤学孩孪孫孬孰孳孵學孺孽宀宁它宄宅宇守安宋完宏宓宕宗官宙定宛宜宝实宠审客宣室宥宦宪宫宰害宴宵家宸容宽宾宿寂寄寅密寇富寐寒寓寕寝寞察寡寤寥實寧寨寫寬寮寰寳寵寶寸对寺寻导対寿封専尃射将將專尉尊對導小尐少尓尔尕尖尘尙尚尛尝尤尥尧尨尬就尴尷尸尹尺尻尼尽尾尿局屁层居屆屈屉届屋屌屎屏屐屑展屛属屠屡層履屯山屹屿岁岂岌岐岑岔岖岗岘岙岚岛岢岩岫岬岭岱岳岷岸岽岿峁峄峋峒峙峡峤峥峦峨峩峪峭峯峰島峻崂崃崆崇崋崎崔崖崛崤崧崩崬崭崮崴崽嵇嵊嵋嵌嵐嵘嵩嵬嵯嶂嶙嶪嶺嶽巅巍川州巡巢巣工左巧巨巩巫差巯己已巳巴巷巽巾币市布帅帆师希帏帐帑帔帕帖帘帚帛帜帝帥带帧師席帮帯帰帶帷常帼帽幂幄幅幌幐幔幕幚幛幡幢幣干平年并幷幸幹幺幻幼幽广庄庆庇床序庐库应底庖店庙庚府庞废庠庥度座庫庭庵庶康庸庹庾廉廊廓廖廚廠廣廰廳延廷建廿开异弃弄弈弊弋弍式弑弓引弗弘弛弟张弥弦弧弩弭弯弱張強弹强弼彈彌彐归当录彗彝形彤彦彧彩彪彬彭彰影彳彷役彻彼往征径待徇很徉徊律後徐徒徕得徘徙徜從御徨復循微徳徵德徹徽心忄必忆忌忍忏忐忑忒忖志忘忙応忠忡忧忪快忱念忸忻忽忾忿怀态怂怄怅怆怎怏怒怔怕怖怜思怞怠怡急怦性怨怩怪怫怯怵总恁恂恃恋恍恐恒恕恙恢恣恤恨恩恪恫恬恭息恰恳恶恸恹恺恻恼恽恿悄悅悉悌悍悒悔悖悚悟悠患悦您悬悯悱悲悴悵悶悸悻悼情惆惊惋惑惕惘惚惜惟惠惡惦惧惨惩惫惬惭惮惯惰想惴惶惹惺愁愈愉愎意愕愚愛感愠愣愤愧愫愿慄慈態慌慎慑慕慢慧慨慰慵慶慷憋憎憔憧憨憩憬憶憾懂懈應懊懋懑懒懦懮懵懷懿戀戈戊戋戌戍戎戏成我戒或战戚戛戟戢截戬戮戳戴戶户戾房所扁扇扈扉手扌才扎扑扒打扔托扙扛扞扣扦执扩扪扫扬扭扮扯扰扳扶批扼找承技抄抉把抑抒抓投抖抗折抚抛抠抡抢护报抨披抬抱抵抹抻押抽抿拂拄担拆拇拈拉拌拍拎拐拒拓拔拖拗拘拙拚招拜拟拢拣拥拦拧拨择括拭拮拯拱拳拴拷拼拽拾拿持挂指按挎挑挖挚挛挝挞挟挠挡挣挤挥挨挪挫振挲挺挽捂捅捆捉捋捌捍捎捏捐捕捞损捡换捣捧据捱捲捶捷捺捻掀掂掇授掉掊掌掏掐排掖掘掠探掣接控推掩措掬掮掰掳掴掷掸掺掼揄揉揍描提插揖揚換握揣揩揪揭援揶揸揽揿搀搁搂搅損搏搐搓搔搜搞搡搧搪搬搭搴携搽摁摄摆摇摈摊摒摔摘摞摧摩摸摹摺撂撃撅撇撑撒撕撞撤撩撬播撮撰撵撷撸撻撼擀擂擅擇操擎擒擘據擞擢擤擦攀攒攘攝攞攥攫支攵收攸改攻放政故效敌敏救敕敖教敛敝敞敟敢散敦敬数敲整敷數文斋斌斐斑斓斗料斛斜斟斡斤斥斧斩斫断斯新方於施旁旅旋旌旎族旖旗无既日旦旧旨早旬旭旮旯旱时旷旸旺旻昀昂昆昇昊昌明昏易昔昕昙昝星映春昧昨昭是昱昴昵昶昼昽显晁時晃晋晌晏晒晓晔晕晖晗晚晞晟晤晦晧晨普景晰晳晴晶晷智晾暂暃暄暇暑暖暗暢暧暨暮暴暸暹曉曙曚曜曝曦曬曰曲曳更書曹曼曾替最會月有朋服朐朔朕朗望朝期朥朦木未末本札术朱朴朵机朽朿杀杂权杆杈杉杋李杏材村杖杜杞束杠条来杨杩杭杯杰東杲杳杵杷松板极构枇枉枋析枕林枚果枝枞枢枣枪枫枭枯枰枱枳架枷枸枹枼柄柏某柑柒染柔柘柚柜柞柠查柩柬柯柰柱柳柴柿栀栅标栈栉栋栎栏树栓栖栗校栢栩株栱样核根格栽栾桀桁桂桃桅框案桉桌桎桐桑桓桔桖桠桡桢档桤桥桦桧桨桩桶桷桼梁梅梆梏梓梗條梢梦梧梨梭梯械梳梵检棂棉棋棍棒棕棘棚棛棟棠棣森棰棱棵棺棽椁椅椋植椎椐椒椤椭椰椴椹椽椿楂楊楓楔楗楚楝楞楠楣楦業極楷楸楹楼楽楿概榄榆榈榉榔榕榛榜榞榧榨榫榭榮榴榷榻槁構槌槍槎槐槑槗様槛槟槲槽槿樂樊樓標樟模樣樨横樫樯樱樵樸樹樽樾橄橇橋橘橙橛機橡橪橱橹橼檀檄檐檔檗檩檫檬檰檸櫥權欠次欢欣欧欲欶欺款歆歇歉歌歐歓歙歡止正此步武歧歪歷歸歹歺死歼殁殃殆殇殉殊残殒殓殖殚殡殴段殷殺殻殼殿毁毂毅毋母每毒毓比毕毖毗毙毛毡毫毬毯毵毽氅氏氐民氓气氖気氘氙氚氛氟氡氢氣氤氦氧氨氩氪氮氯氰氲水氵永氽汀汁求汆汇汉汊汐汕汗汛汝汞江池污汤汨汩汪汰汲汴汶汹決汽汾沁沂沃沅沈沉沌沏沐沓沔沙沛沟没沣沤沥沦沧沪沫沭沮沱河沸油治沼沽沾沿泄泅泉泊泌泓泔法泖泗泛泞泠泡波泣泥注泩泪泫泮泯泰泱泳泵泷泸泺泻泼泽泾洁洄洋洒洗洙洛洞津洨洪洮洱洲洳洵洺活洼洽派流浃浅浆浇浈浊测浍济浏浐浑浒浓浔浙浚浜浠浣浥浦浩浪浮浯浴海浸浼涂涅涇消涉涌涎涑涓涔涕涛涝涞涟涠涡涣涤润涧涨涩涪涫涮涯液涵涸涿淀淄淅淆淇淋淌淑淖淘淙淚淝淞淡淤淦淨淩淫淬淮深淳混淸淹添淼渄清渊渌渍渎渐渑渔渗渘渚減渝渠渡渣渤渥温測渭港渲渴游渺湃湄湊湍湎湓湖湘湛湟湫湮湯湾湿溃溅溆溇溉溏源溜溝溟溢溥溧溪溫溯溱溴溶溺滁滂滇滋滏滑滓滔滕滘滙滚滞滟满滢滤滥滦滨滩滬滴滿漁漂漆漉漏漓演漕漟漠漢漩漪漫漯漱漳漸漾潇潋潍潔潘潜潞潢潤潦潭潮潴潸潺潼澄澈澍澎澜澡澤澧澳澶澹激濂濃濉濑濒濕濛濞濟濠濡濤濮濯濱瀑瀘瀚瀛瀣瀵瀹灌灏灞灡灣火灬灭灯灰灵灶灸灼灾灿炀炅炉炊炎炒炔炕炖炙炜炝炣炫炬炭炮炯炳炸点為炻炼炽烁烂烃烈烊烏烘烙烛烜烟烤烦烧烨烩烫烬热烯烷烹烺烽焉焊焐焕焖焗焘焙焚無焦焯焰焱然焼煅煊煌煎煒煖煙煜煞煤煥煦照煨煮煲煳煸煽熄熊熏熔熘熙熟熠熥熨熬熱熳熵熹燁燃燈燉燎燒燕燘燙燚燜營燥燦燧燮燴爆爍爐爨爪爬爭爯爰爱爲爵父爷爸爹爺爻爽爾爿片版牌牍牒牙牛牝牟牡牢牦牧物牯牲牵特牺犀犁犄犇犊犍犒犟犬犭犯犴状犷犸犹狀狂狄狈狍狐狒狗狙狞狠狡狩独狭狮狰狱狲狸狺狼猁猎猕猖猗猛猜猝猞猥猩猪猫猬献猴猷猾猿獅獐獒獗獠獨獭獲獴獸獻獾玄率玉王玎玑玖玛玟玢玥玩玫玮环现玲玳玷玺玻珀珂珈珉珊珍珏珐珑珙珞珠珣珥珩班珲珺現球琅理琉琊琏琐琚琛琢琤琥琦琨琪琬琮琯琰琳琴琵琶琼瑁瑄瑅瑕瑗瑙瑚瑛瑜瑞瑟瑩瑪瑭瑰瑶瑷瑾璀璁璃璇璈璋璎璐璜璞璟璧璨璩環璺瓒瓜瓠瓢瓣瓤瓦瓮瓯瓴瓶瓷瓿甄甘甙甚甜生產産甥用甩甫甬甭甯田由甲申电男甸甹町画甾畅畈界畏畐畔留畜略畦番畫畲畴畵當畸畹畿疃疆疌疏疑疖疗疙疚疝疟疡疣疤疥疫疮疯疱疲疳疵疸疹疼疽疾痂病症痈痉痊痍痒痔痕痘痛痞痠痢痣痤痧痨痪痫痰痱痴痹痼痿瘀瘁瘊瘋瘘瘙瘟瘠瘢瘤瘦瘩瘪瘫瘴瘸瘾瘿療癌癔癖癜癞癣癫癸登發白百皂的皆皇皈皋皎皑皓皖皙皮皱皲皴皺皿盂盅盆盈益盎盏盐监盒盔盖盗盘盛盜盟盤盥目盯盱盲直相盹盼盾省眈眉看眙眞真眠眦眨眩眬眭眯眶眷眸眺眼着睁睇睐睑睛睡睢督睦睨睫睬睹睽睾睿瞀瞄瞅瞌瞎瞑瞒瞓瞟瞠瞥瞧瞩瞪瞬瞭瞰瞳瞻瞿矍矗矛矜矞矢矣知矩矫短矮石矶矸矽矾矿砀码砂砋砌砍砒研砖砚砜砝砣砥砦砧砬砭砰破砷砸砺砻砼砾础硂硅硌硒硕硖硚硝硪硫硬确硷硼碁碇碉碌碍碎碑碓碗碘碚碜碟碡碣碧碩碰碱碲碳碴碶確碼碾磁磅磊磋磐磕磙磚磨磬磳磴磷磺礁礴示礻礼社祀祁祇祈祉祎祐祖祗祙祚祛祜祝神祟祠祢祥票祭祯祷祸祺祼禀禁禄禅福禓禦禧禪禮禹禺离禽禾秀私秃秆秉秋种科秒秕秘租秣秤秦秧秩秫秭积称秸移秽稀程稍税稔稗稚稞稠稣種稳稷稹稻稼稽稿穂穆穗穩穰穴穵究穷穸穹空穿突窃窄窈窍窑窒窕窖窗窘窜窝窟窠窣窥窦窨窸窿立竖站竞竟章竣童竭端競竹竺竽竿笃笆笈笊笋笏笑笔笕笙笛笞笠笤符笨笫第笺笼筆等筋筏筐筑筒答策筛筝筠筱筲筳筵筷筹签简箍箐箔箕算管箤箩箫箬箭箱箴箸節篁範篆篇築篑篓篙篝篡篦篪篮篱篷篾簇簋簌簏簡簧簪簸簽簿籁籃籍籠籣籤米类籼籽粉粑粒粕粗粘粝粞粟粢粤粥粧粪粮粱粲粳粵粹粼粽精粿糁糅糊糍糕糖糗糙糜糝糟糠糬糯系紀約紅紊紋納紐純紗級素紡索紧紫累細紹終組結絡給絨絮統絲絶綁綉經綦綫維綱網緑緖線緣編緯緹緻縣縤縫縮總繁織繡繪繹纂續纛纟纠红纣纤约级纨纪纫纬纭纯纰纱纲纳纵纶纷纸纹纺纽纾线绀绁绂练组绅细织终绉绊绋绌绍绎经绐绑绒结绔绕绗绘给绚绛络绝绞统绠绡绢绣绥绦继绨绩绪绫续绮绯绰绱绲绳维绵绶绷绸绺绻综绽绾绿缀缁缂缃缄缅缆缇缈缉缍缎缐缓缔缕编缘缙缚缜缝缠缢缤缥缦缨缩缪缫缬缭缮缰缴缸缺缽罂罄罅罐网罔罕罖罗罘罚罡罢罩罪置罱署罹羁羅羊羌美羔羙羚羞羟羡群羧義羮羯羰羲羸羹羽羿翁翅翊翌翎翏習翔翘翟翠翡翥翦翩翰翱翳翻翼耀老考耄者耆而耍耐耑耒耕耗耘耙耜耦耧耪耱耳耵耶耷耸耻耽耿聂聆聊聋职聒联聖聘聚聞聨聩聪聯聲職聾聿肃肄肆肇肉肋肌肓肖肘肚肛肝肟肠股肢肤肥肩肪肫肮肯肱育肴肺肼肽肾肿胀胁胃胄胆背胍胎胖胗胚胛胜胞胡胤胥胧胪胫胬胭胯胰胱胳胴胶胸胺胼能脂脆脉脊脍脏脐脑脒脓脖脘脚脯脱脲脸脾腆腈腊腋腌腎腐腑腓腔腕腚腥腦腧腩腭腮腰腱腴腸腹腺腻腼腾腿膀膈膊膏膑膓膘膚膛膜膝膠膦膨膳膺膻臀臂臃臆臊臘臜臣臥臧臨自臭至致臺臻臼臾舀舅舆與興舊舌舍舐舒舔舖舛舜舞舟航舫般舰舱舵舶舷舸船艄艇艘艦艮良艰色艳艶艷艹艺艾艿节芃芊芋芍芎芒芗芘芙芜芝芡芥芦芨芩芪芫芬芭芮芯花芳芷芸芹芽芾苁苄苇苋苍苎苏苑苒苓苔苕苗苘苛苜苝苞苟苡苣若苦苪苫苯英苴苷苹苺苼茁茂范茄茅茆茉茌茎茏茑茔茗茛茜茧茨茫茬茭茯茱茴茵茶茸茹茼茾荀荃荆草荏荐荒荔荘荚荞荟荠荡荣荤荥荧荨荪荫荬荭药荷荸荻荼荽莅莆莉莊莎莒莓莘莜莞莠莨莪莫莱莲莳莴获莹莺莼莽菀菁菅菇菊菌菏菓菖菘菜菟菠菡菩菪菮華菱菲菴菸菽萁萃萄萆萊萋萌萍萎萏萘萜萝萤营萦萧萨萬萭萱萸萼落葆葉葐葑葒著葙葚葛葡董葩葫葬葭葱葳葵葸葺蒂蒄蒋蒌蒙蒜蒟蒡蒤蒲蒸蒺蒻蒽蒿蓄蓉蓋蓑蓓蓖蓝蓟蓥蓦蓬蓮蓼蓿蔃蔑蔓蔗蔚蔟蔡蔫蔬蔷蔸蔺蔻蔼蔽蕃蕈蕉蕊蕌蕒蕙蕟蕤蕨蕲蕴蕹蕾薄薇薈薏薑薛薡薦薩薪薬薮薯薰薹藁藉藍藏藐藓藔藕藜藝藤藥藩藻藿蘆蘇蘑蘖蘭蘸蘼蘿虎虏虐虑虔處虚虛虞號虢虫虬虭虱虹虻虽虾蚀蚁蚂蚊蚋蚌蚓蚕蚜蚝蚣蚤蚧蚨蚩蚪蚬蚯蚱蚴蚵蚶蛀蛆蛇蛊蛋蛎蛏蛐蛔蛙蛛蛟蛤蛭蛮蛰蛲蛳蛴蛹蛾蜀蜂蜃蜇蜈蜉蜊蜍蜒蜓蜕蜗蜘蜚蜛蜜蜞蜡蜢蜣蜥蜱蜴蜷蜻蜿蝇蝈蝉蝌蝎蝗蝙蝠蝣蝦蝮蝰蝴蝶蝽蝾螂螃螅螈螋融螟螥螨螫螯螳螺蟀蟆蟊蟋蟑蟒蟛蟠蟹蟾蠄蠓蠔蠕蠖蠡蠢蠹蠼血衅衆行衍術衔衖街衙衛衡衢衣补表衩衫衬衮衰衲衷衾衿袁袂袄袅袆袈袋袍袒袖袜袤袪被袭袱裁裂装裆裔裕裘裙補裝裟裡裢裤裥裨裱裳裴裸裹裼製裾褀褂複褐褒褓褔褚褛褡褥褪褰褲褴褶褸襁襄襞襟西要覃覆見覌視親覺觀见观规觅视觇览觉觊觎觐觑角觞解触觸言訂訇計訊訓記訣訪設許診訾詢試詩話該詹誉誊誌誓誘語誠説課調請諾謇講謝謢謹證識譜警譬護讀變讠计订讣认讥讨让讪讫训议讯记讲讳讴讶讷许讹论讼讽设访诀证诃评诅识诈诉诊诋诌词诏译诒诓试诗诘诙诚诛话诞诟诠诡询诣诤该详诧诩诫诬语误诰诱诲说诵诶请诸诺读诽课诿谀谁调谄谅谆谈谊谋谌谍谎谏谐谑谒谓谔谕谗谘谙谚谛谜谟谡谢谣谤谥谦谧谨谩谪谬谭谯谰谱谲谴谷豁豆豇豉豊豌豐豕豚象豢豪豫豬豹豺貂貅貉貊貌貔貝負財貢貨販貭貴買費貼貿賀資賓賜賞賠賢賣質賴購贈贏贝贞负贡财责贤败账货质贩贪贫贬购贮贯贰贱贲贴贵贷贸费贺贻贼贾贿赁赂赃资赅赈赉赊赋赌赎赏赐赓赔赖赘赚赛赝赞赠赡赢赣赤赦赧赫赭走赳赴赵赶起趁趄超越趋趔趙趟趣趱足趴趵趸趾趿跃跄跆跋跌跎跑跖跚跛距跟跤跨跪路跳践跶跷跹跺跻踅踉踊踌踏踝踞踟踢踨踩踪踮踯踱踵踹踺踽蹂蹄蹇蹈蹉蹊蹋蹑蹒蹓蹙蹚蹦蹩蹬蹭蹰蹲蹴蹶蹼蹿躁躅躇躏身躬躯躲躺車軌軍軎軒軟軽載輕輝輪輯輸轃轉车轧轨轩转轭轮软轰轱轲轳轴轶轻轼载轿较辄辅辆辈辉辊辋辍辐辑输辔辕辖辗辘辙辛辜辞辟辣辦辧辨辩辫辰辱農边辽达迁迂迄迅过迈迎运近返还这进远违连迟迢迤迥迦迩迪迫迭迮述迳迷迸迹追退送适逃逄逅逆选逊逍透逐逑递途逗這通逛逝逞速造逡逢連逦逭逮逯進逵逶逸逹逺逻逼逾遁遂遇運遍過遏遐遒道達遗遛遠遢遣遥遨適遭遮遴遵選遽避邀邂邃還邈邋邏邑邓邕邗邛邝邡邢那邦邨邪邬邮邯邰邱邳邴邵邸邹邺邻郁郄郅郊郎郏郑郓郗郛郜郝郡郢郦郧部郫郭郯郴郵郸都郾鄂鄄鄉鄙鄞鄢鄣鄭鄯鄱鄺酉酊酋酌配酐酒酗酚酝酞酡酢酣酥酩酪酬酮酯酰酱酵酶酷酸酽酿醇醉醋醌醍醐醒醚醛醜醪醫醬醯醴醺釀采釉释里重野量金釜針釣鈴鈺鈽鉄鉅鉌鉴銀銑銘銭銮銷鋒鋪鋼錄錢錦錯錶錾鍊鍋鍵鍾鎏鎖鎢鎮鎻鏖鏜鏡鏢鐘鐡鐵鑒鑫鑽钅钇针钉钊钎钏钒钓钕钗钙钚钛钜钝钞钟钠钡钢钣钥钦钧钨钩钬钮钯钰钱钲钳钴钵钹钺钻钼钽钾钿铀铁铂铃铄铅铆铈铉铊铋铌铍铎铐铑铒铔铕铖铗铛铜铝铞铟铠铡铢铣铤铧铨铪铬铭铮铰铱铲铵银铸铺链铿销锁锂锃锄锅锆锈锉锋锌锏锐锑锒锔锕锗锘错锚锛锝锟锡锢锣锤锥锦锨锭键锯锰锲锴锵锶锷锹锺锻锽镀镁镂镇镉镊镌镍镐镑镒镓镔镖镗镘镛镜镝镢镣镦镧镩镫镭镯镰镱镳镴镶長长門閃開閏閑間閟関閣閥閩閮關门闩闪闫闭问闯闰闱闲闳间闵闷闸闹闺闻闽闾闿阀阁阂阄阅阆阈阉阊阍阎阐阑阒阔阕阖阗阙阚阜阝队阡阪阮阱防阳阴阵阶阻阿陀陂附际陆陇陈陉陋陌降限陕陛陟陡院陣除陨险陪陰陲陳陵陶陷陽隅隆隋隍随隐隔隗隘隙際障隣隧隨隶隹隻隼隽难雀雁雄雅集雇雉雌雍雎雏雒雕雙雜雞離難雨雪雯雲雳零雷雹電雾需霁霄霆震霉霍霎霏霓霖霜霞霦霭霰露霸霹霾靈靌靑青靓靖静靛靜非靠靡面靥革靳靴靶鞅鞋鞍鞑鞘鞠鞣鞦鞭鞯韓韦韧韩韫韬韭音韵韶韻響頂順頓頔頡頤頭頰題額顏顔類顧顯页顶顷项顺须顽顾顿颀颁颂预颅领颇颈颉颊颌颍颏颐频颓颔颖颗题颚颛颜额颞颠颡颢颤颦颧風飄风飏飒飓飕飘飙飚飛飞食飨飬飮飯飲飼飾餃餅養餐餓餘餠館餮饃饅饕饣饥饦饨饪饬饭饮饯饰饱饲饴饵饶饷饸饹饺饼饽饿馀馁馄馅馆馈馊馋馍馏馐馒馔馕首馗香馥馨馬馮馴駕駿騰騷驊驗马驭驮驯驰驱驳驴驶驷驸驹驻驼驽驾驿骁骂骄骅骆骇骊骋验骏骐骑骓骗骚骛骜骝骞骠骡骤骥骧骨骰骶骷骸骺骼髁髂髅髋髌髓體高髙髦髪髮髯髻鬃鬄鬈鬓鬟鬣鬲鬼魁魂魄魅魇魉魏魑魔魖魚鮀鮑鮣鮮鯉鯛鯨鯽鰤鰻鱈鱔鱬鱸鱻鱼鱿鲁鲂鲅鲆鲇鲈鲍鲑鲛鲜鲞鲟鲢鲤鲨鲩鲫鲭鲮鲱鲲鲳鲵鲶鲷鲸鲽鳃鳄鳅鳌鳍鳎鳏鳐鳕鳖鳗鳙鳜鳝鳞鳟鳥鳯鳳鴒鴝鴨鴻鴿鵬鵳鶏鶴鷄鷉鷗鷹鷺鸊鸟鸠鸡鸢鸣鸥鸦鸨鸪鸫鸬鸭鸯鸲鸳鸵鸶鸽鸾鸿鹀鹂鹃鹄鹅鹈鹉鹊鹌鹏鹐鹑鹗鹘鹚鹜鹞鹤鹦鹧鹩鹪鹫鹬鹭鹮鹰鹳鹽鹿麂麋麒麓麗麝麟麥麦麯麴麵麸麹麺麻麼麽麾黃黄黍黎黏黑黒黔默黛黜黝點黟黠黢黨黯鼎鼓鼙鼠鼢鼬鼯鼹鼻鼽鼾齊齋齐齿龃龄龇龈龉龊龋龌龍龙龚龛龟龢郎凉﹑﹗﹝﹞﹢！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？ＡＢＣＤＥＦＧＨＩＫＬＭＮＯＰＲＳＴＵＶＷＹＺ［］｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｒｓｔｕｚ｛｜｝～￠￡￥𣇉"
head = SRNHead(charsets)
conv_inputs = torch.rand((16,512,8,int(256/8)))
b = head(dict(img=conv_inputs),return_loss=False)


import torchvision.models as models

models
# slf_attn_layer = MultiheadAttention(embed_dim=512,num_heads=8)
# b = position_embedding_layer(conv_features)
# #[16,256,512],[256,16,16]
# a,c=slf_attn_layer(b,b,b)
# l = torch.nn.LayerNorm(512)
# d = l(a)
# pos_ffn_layer = PositionwiseFeedForward(512, 512, dropout=0.1)
# out = pos_ffn_layer(d)
