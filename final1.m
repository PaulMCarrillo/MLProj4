clear;close;clc;

[x1,report] = import_worm_data();
X = x1';
[coeff, score, ~, ~, explained, ~] = pca(X,'Rows','all', 'Algorithm', 'eig');
A1 = score(:,1:200);

A1 = [A1 A1.^2];
A1 = [A1 ones(size(A1,1),1)];

weight = [-7163.34197204138; 1078.47786331253; 5626.68980780386; 132.433276599405; 1590.51809402226; -85.5704775323942; -977.087992482682; -93.9090459916606; -3758.04449250346; 2108.85324578782; 3617.20672747416; 933.431059415275; 233.023544402399; -446.845865635056; 305.536070624876; -1520.42471229402; 574.339236481761; -197.037109262654; 738.404995049728; 255.706605558958; -407.625388599201; -518.657905976029; -89.0605259644402; -709.648982432297; 348.954462449655; 683.623444392258; -28.0015511449949; -132.973464598608; 1241.53918342894; -164.951406851215; -381.743247354189; 776.108973011060; -522.535924333884; -239.302397218725; 2.71131247922960; 97.4185396798247; 926.422189327531; -355.131694088139; -749.703573222451; 154.185550401069; 417.674041844785; -263.195942559267; 265.114104918913; -192.546809800653; 61.0603995443710; -100.621412556505; -486.822364752537; -996.892893521461; 154.020405323420; -631.490551846973; -357.646935258643; 862.539223319999; -651.589988366213; 165.968942100957; 90.7764984001897; -59.4295573893131; -187.791676841906; -198.655421912409; -243.710729335306; -56.6419260062941; -690.053747154128; 1040.55604937492; 639.936514443837; -219.244428454054; 721.165046540263; 190.511210780802; -390.610318117012; -566.646981475101; -719.078381490125; 186.173773184257; 151.461440799017; -11.9303760822943; -29.0439946727827; -105.551298230578; -249.640862226491; 837.224621868244; 146.673961948443; 305.487990123046; -403.471958243301; -730.686108388410; 45.2907286167136; 251.327449112570; -267.942872412317; 443.803386282769; -245.757118800415; -503.565083846638; 3.98673241679992; -409.216020170203; -701.724991411819; -49.9479871307245; 625.702443326312; 55.6750193023568; 299.584601525698; 693.045522194364; -10.9750148970806; -32.3732807525183; 130.571118271992; -380.315309543013; -149.603106762741; -743.589315833836; 8.36671566599216; -788.630545727373; -333.312276056363; -396.166896408426; -425.610593855818; 104.905670112156; 68.2861806088318; -536.018749574082; 202.544900879573; -82.0841823296850; 256.078981285534; -62.5156197392645; -20.8743392477884; -200.073868021738; 22.0530198059872; 116.547460709796; 90.9928199907686; 270.050818457536; 170.296197714494; 363.358898883008; 265.822569272913; 317.655318784766; -118.233554948495; -566.367884707161; 147.371544574333; -581.105583359771; -146.781362995016; 506.827953221766; -2.17225005008174; -388.116978540156; -99.3027614038423; 386.058101945687; -76.2261695986414; -87.4629449880936; -149.960946354071; 43.1467009273532; -252.959344240613; -20.8949526669316; 120.819103943011; -257.831318729735; 73.7435820544409; 63.0921665941978; -199.999571432047; -260.112726223466; 134.060410838822; 244.755976336655; -37.0729221501314; -58.8644412831850; -276.142227124426; -92.9444037057933; -204.424814658683; 250.294039680289; -297.895055780462; -380.483658173433; -93.8658821457299; -136.166361574055; -154.018908988007; 34.3876363042677; -528.201782338112; 228.758006627357; -140.368853540997; -280.678045978698; 119.408354092775; -92.0645280838834; -219.036151905198; 20.7288345971734; -79.6297498242061; -206.005880507008; -36.6151526087898; -214.993757232559; 3.68203434795896; 118.409192120409; -25.7527778101743; 191.216304116559; 98.4753474739306; 81.4032820342521; 196.149561966350; 25.2388657397663; -148.839686925677; 181.157206223456; 302.333114203306; -251.385417002334; 79.8155208456040; -106.685952931910; -244.482869909733; 31.4591607432589; -57.1612996204081; -4.93261387607604; 2.85611626594245; -115.301718144060; -176.000196710682; 269.796104997243; 193.380331475242; 67.6130937851146; -115.837919652668; -37.6787314678296; -184.397711233936; -44.3864792974964; 139.788502571097; 59.1081737947208; 145.687979227173; -57.5214044737384; -2574.51061153860; -33.9622488361401; -299.006043346623; 259.371062063335; -1897.94197381067; -786.381952966479; -367.810920736811; -959.137041702758; -981.793383389976; -115.939232468444; 1404.51185391406; 903.428921459999; -389.071126070457; 81.6343714955818; -572.893234897827; -2307.65236730523; 389.825920486855; 801.724239826940; 2161.24269142461; 1554.46154591655; 1135.31467551654; 1967.90469535015; -256.996919948631; 536.150708906742; 798.604104923986; 77.7483573258982; 521.613514021636; 586.215882221780; 517.708744860101; 48.8343298276374; 1164.06455078100; 2088.57132305364; 1544.15520849289; -99.4228003520762; 1120.53666553766; 1548.52536036286; 1648.49738507722; 1309.14136576723; 63.8015201550216; -659.270791747555; 1068.01816877013; 508.963109281691; 706.416746015564; 241.231528471541; 446.112783496244; 269.758753653011; 929.831364784761; 542.477199060268; -436.497688560537; 1061.36357138884; 567.648338786639; 435.624744259354; 471.116410028772; 1129.44898109710; 1134.20409555454; 530.488318374650; 710.693114675980; 1694.83747950669; 615.383912704397; 296.285741066797; 112.969381526128; 937.741118756970; 751.501652822746; 21.4490078423125; 192.060140067742; 284.390368649585; 558.921786195123; 762.122116516873; 122.672261538201; 130.584597073866; 255.271794687484; 411.282980507862; -34.2605672932260; -190.829359756458; -178.179788156889; 309.470991983567; 176.872312512494; 321.096088693011; 98.9938785107340; 425.800192033894; 335.241359212033; 365.501538243936; 310.506908535733; 143.563322481559; -597.483387261902; 491.979769143093; 52.4134769016931; -60.0308632692689; 161.828088340393; 465.334071311290; 332.001447252303; 731.267990693393; 401.886619876002; 311.488466039040; 227.721685625393; -195.553662130548; 312.987730352233; 202.234303338963; 79.0403068852124; -206.023969901290; 102.189310652195; 103.554054921071; 455.859395156641; -28.7428077346827; 261.923868195767; -204.481050664638; 124.444727358634; 103.697227773285; 145.376544730258; 294.002011291612; 234.047851308294; -233.353296931137; 4.26728573259818; 313.887840450181; 159.710243153896; -88.3687859611617; 33.7546727605988; -86.6319893296809; 5.64764836881254; -94.3068815618808; 42.6976230023019; -83.0381296981445; -138.223822799214; -21.1774701711533; -84.8880255926839; -234.723919306546; 99.0111830724346; -156.969022002859; -1.72181809059159; -14.7378112555472; 113.287021433253; -146.278311990109; -0.602197348148796; 292.554020751945; -59.3546036232859; 118.410849472353; -334.815890120509; -175.188631181851; -257.267287340209; -183.538333085986; -63.6477345350236; 156.288321889206; -41.4518472742929; -40.3092642307008; -202.359986787091; -59.3302416297594; -84.4695490991155; -425.439282950320; -124.443397264563; -358.543932361140; -151.197978840082; -106.674384417796; -9.00940374760041; -126.059755928755; -144.811807674972; -63.6811845979459; -164.483161509676; -28.1345170576266; -278.191424008450; -119.768785830656; -242.768920261526; -16.6405556175643; -227.211707785386; -290.428526716893; -53.1610749920174; -124.442094463685; -267.812949757909; -147.291876631119; -179.330128911571; 22.5215274409052; -316.735595166984; -158.046505496434; -206.667339994904; -69.7674581977167; -257.263609739275; -70.7195433515980; -289.428955568177; -46.5897775532845; -132.395298210862; -307.575972201634; -166.689844366929; -161.013919816316; -204.592919381947; -417.093561472584; -145.044709361447; -84.8583878336048; -175.500232042982; -279.407119030358; -304.407907084185; -118.049683619562; -131.967244686144; -46.1787984092107; -108.213048008476; -166.967179078922; -152.871154004887; -215.063641138821; -114.706783252945; -153.426824518819; 588.187600798926];


y1 = A1*weight;
y2 = sigmoid_activate(y1');
y3 = classify(y2);

count0 = 0;
count1 = 0;
l = length(y3);
for i = 1:l
    
    report(i,2) = y3(i);
    
    if(y3(i) == 1)
        count1 = count1 + 1;
    else
        count0 = count0 + 1;
    end
end

report(l+1,1) = "Total 1's";
report(l+1,2) = count1;
report(l+2,1) = "Total 0's";
report(l+2,2) = count0;

exportfile='results.xlsx';
xlswrite(exportfile,report);

export_model = 'model.xlsx';
xlswrite(export_model,weight);

function [worm_data, result] = import_worm_data()
    
    prompt = 'Enter image directory name: ';
    image_folder = input(prompt,'s');
    image_files=dir([image_folder '/*.png']); %Images are in png format.
    L = length(image_files);
    val = 0.36;
    sz = ceil(101*val);
    g = sz*sz;
    worm_data = zeros(g, L);
    result = strings(L+2,2);
    for k=1:L
      filenames=[image_folder '/' image_files(k).name];
      baseFileName = image_files(k).name;
      result(k,1) = baseFileName;
      fullFileName = fullfile(image_folder, baseFileName);
      data = imread(fullFileName);
      cval_t = imresize(data,val);
      cval_t = imbinarize(cval_t);
      re_cval = cval_t(:)'; 
      
      worm_data(:, k) = re_cval;
    end
    
end



function [y] = sigmoid_activate(x_entries)
    y = zeros(1,size(x_entries, 2));
    for i = 1:size(x_entries, 2)
        answ = 1/(1 + exp(-x_entries(1, i)));
        y(1, i) = answ;
    end
end


function [y] = classify(y_hat)
    y = zeros(1, size(y_hat, 2));
    for i = 1:size(y_hat, 2)
        if y_hat(1, i) >= 0.5
            y(1, i) = 1;
        end
        if y_hat(1, i) < 0.5
            y(1, i) = 0;
        end
    end   
end







