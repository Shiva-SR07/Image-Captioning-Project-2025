Image Captioning-Project-2025
Some of the core Concepts that I worked is 
 sharing with you all.

1. processing Images & Captions data

2. Extracting image features via CNN

3. Training the LSTM to generate Captions 

4. By using the Trained model we can predict 
 Captions for new images.

5. Finally our project is ready for Testing by 
 providing images it will generate Captions as 
 description.
1)Terminal commands to run the Project: I mean to if you paste these random commands in your terminal it will give you the image along with Captions
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\3765374230_cb1bbee0cb.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\3767841911_6678052eb6.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\377872472_35805fc143.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\377872672_d499aae449.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\378170167_9b5119d918.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\392467282_00bb22e201.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\392976422_c8d0514bc3.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\393284934_d38e1cd6fe.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\393810324_1c33760a95.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\393958545_48c17c66d1.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\393987665_91d28f0ed0.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\424779662_568f9606d0.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\424869823_7aec015d87.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\425088533_a460dc4617.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\425518464_a18b87c563.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\469021173_aa31c07108.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\469029994_349e138606.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\469259974_bb03c15c42.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\469617651_278e586e46.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\469969326_4b84073286.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\470373679_98dceb19e7.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\470887781_faae5dae83.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\470887785_e0b1241d94.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\470887791_86d5a08a38.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\489065557_0eb08889cd.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\489134459_1b3f46fc03.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\489372715_ce52da796a.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\489551372_b19a6ad0ed.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\489773343_a8aecf7db3.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\489865145_65ea6d1c14.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\490044494_d2d546be8d.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\490390951_87395fcb1c.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\491405109_798222cfd0.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\491564019_1ca68d16c1.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\491600485_26c52c8816.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\491964988_414b556228.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\492341908_1ef53be265.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\492493570_c27237a396.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\492802403_ba5246cfea.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\493109089_468e105233.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\496129405_b9feeda1ab.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\496380034_d22aeeedb3.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\496555371_3e1ee0d97d.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\496606439_9333831e73.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\496971341_22782195f0.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\497122685_a51b29dc46.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\497579819_f91b26f7d3.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\497791037_93499238d8.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\498404951_527adba7b8.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\498444334_a680d318a1.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\498492764_fe276e505a.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\498748832_941faaaf40.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\498794783_cc2ac62b47.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\498957941_f0eda42787.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\50030244_02cd4de372.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\500308355_f0c19067c0.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\500446858_125702b296.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\500678178_26ce0f4417.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\501320769_31eea7b7ea.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\501520507_c86f805ab8.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\501650847_b0beba926c.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\501684722_0f20c4e704.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\501699433_f8df386cf9.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\502115726_927dd684d3.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\502671104_b2114246c7.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\502783522_3656f27014.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\502884177_25939ac000.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\503090187_8758ab5680.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\503717911_fc43cb3cf9.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\503794526_603a7954d3.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\504385521_6e668691a3.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\504765160_b4b083b293.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\504904434_889f426c6e.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\513269597_c38308feaf.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\697490420_67d8d2a859.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\697582336_601462e052.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\697778778_b52090709d.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\698107542_3aa0ba78b4.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\700884207_d3ec546494.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\701816897_221bbe761a.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\707941195_4386109029.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\707972553_36816e53a2.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\708860480_1a956ae0f7.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\708945669_08e7ffb9a7.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\709373049_15b8b6457a.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\70995350_75d0698839.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\710878348_323082babd.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\716597900_b72c58362c.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\717673249_ac998cfbe6.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\719837187_3e7bf1d472.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\720208977_f44c2bba5b.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\72218201_e0e9c7d65b.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\724702877_f2a938766b.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\72964268_d532bb8ec7.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\732468337_a37075225e.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\733172023_5810350af6.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\733752482_ee01a419e5.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\733964952_69f011a6c4.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\733965014_1a0b2b5ee9.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\735787579_617b047319.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\742073622_1206be8f7f.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\743571049_68080e8751.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\745563422_f4fa7d9157.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\745880539_cd3f948837.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\745966757_6d16dfad8f.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\746787916_ceb103069f.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\747242766_afdc9cb2ba.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\747921928_48eb02aab2.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\749840385_e004bf3b7c.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\750196276_c3258c6f1b.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\751074141_feafc7b16c.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\751109943_2a7f8e117f.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\751737218_b89839a311.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\752052256_243d111bf0.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\753285176_f21a2b984d.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\753578547_912d2b4048.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\754852108_72f80d421f.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\755326139_ee344ece7b.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\756004341_1a816df714.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\756521713_5d3da56a54.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\756909515_a416161656.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\757046028_ff5999f91b.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\757133580_ba974ef649.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\757332692_6866ae545c.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\758921886_55a351dd67.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\759015118_4bd3617e60.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\760138567_762d9022d4.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\760180310_3c6bd4fd1f.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\762947607_2001ee4c72.jpg" --feature_model vgg1
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\887108308_2da97f15ef.jpg" --feature_model vgg16
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\888425986_e4b6c12324.jpg" --feature_model vgg1
1)Best picture....
python main.py --mode predict --image_to_predict "D:\Image Captioning\data\images\380034515_4fbdfa6b26.jpg" --feature_model vgg16
