## Approach

This file describes the approach I took for the solution.

As, this is a dataset which has many folders in which the actual images reside there is a great chance to solve the task in parallel.
Now the choice comes between multi-threading or processing and it depends on what is the type of the task is it CPU bound or IO bound.

After seeing those timeit results(below) it was quite conclusive that it is a CPU bound task.
```
python -m timeit -s "import face_recognition" -n 300 "pic=face_recognition.load_image_file('/home/ishanbhatt/Celeb_Test_Images/CR3.jpg')"
	300 loops, best of 3: 15.9 msec per loop
python -m timeit -s "import face_recognition;pic=face_recognition.load_image_file('/home/ishanbhatt/Celeb_Test_Images/CR3.jpg')" -n 20 "encodings = face_recognition.face_encodings(pic)"
	20 loops, best of 3: 795 msec per loop
```

So, I chose multiprocessing to do the stuff.
In that I had 2 choices as well

1. Go for normal multiprocessing with apply_async.
2. Concurrent futures.

I chose concurrent futures because it is easier and very self explanatory.
Submit the tasks in the pool and get results as completed.

**It's critical that you choose number of processors wisely.
A low number would hamper parallelism and a very higher number would not guarantee a great performance.
I chose whatever processors available - 1.**

I also added tqdm in the mix so that user gets the idea how much processing is done.

Also, there were some images for which face_encodings could not find face so returned empty encoding vector. I have ignored such images from our average calculations.
Following are those images.

	Dale_Earnhardt_Jr/Dale_Earnhardt_Jr_0003.jpg IGNORING
	James_McGreevey/James_McGreevey_0002.jpg IGNORING
	Thomas_Birmingham/Thomas_Birmingham_0002.jpg IGNORING
	John_Thune/John_Thune_0001.jpg IGNORING
	Jane_Fonda/Jane_Fonda_0002.jpg IGNORING
	Damarius_Bilbo/Damarius_Bilbo_0001.jpg IGNORING
	Jerry_Rice_0001.jpg IGNORING
	George_HW_Bush/George_HW_Bush_0002.jpg IGNORING
	John_Burkett/John_Burkett_0001.jpg IGNORING
	Billy_Andrade/Billy_Andrade_0001.jpg IGNORING
	Joe_Vandever/Joe_Vandever_0001.jpg IGNORING
	Franz_Muentefering/Franz_Muentefering_0003.jpg IGNORING
	Colin_Montgomerie/Colin_Montgomerie_0004.jpg IGNORING
	Claudia_Pechstein/Claudia_Pechstein_0005.jpg IGNORING
	Philippe_Noiret/Philippe_Noiret_0002.jpg IGNORING
	Harvey_Wachsman/Harvey_Wachsman_0001.jpg IGNORING
	Jon_Stewart/Jon_Stewart_0001.jpg IGNORING
	Luis_Horna/Luis_Horna_0002.jpg IGNORING
	Arlen_Specter/Arlen_Specter_0003.jpg IGNORING
	Don_Hewitt/Don_Hewitt_0001.jpg IGNORING
	Claudette_Robinson/Claudette_Robinson_0001.jpg IGNORING
	Jeff_Feldman/Jeff_Feldman_0001.jpg IGNORING
	Sean_Patrick_OMalley/Sean_Patrick_OMalley_0003.jpg IGNORING
	Donald_Rumsfeld/Donald_Rumsfeld_0115.jpg IGNORING
	Nabil_Shaath/Nabil_Shaath_0002.jpg IGNORING
	Gary_Bald/Gary_Bald_0001.jpg IGNORING
	Jennifer_Capriati/Jennifer_Capriati_0031.jpg IGNORING
	Abdoulaye_Wade/Abdoulaye_Wade_0003.jpg IGNORING
	Saeb_Erekat/Saeb_Erekat_0002.jpg IGNORING
	Michael_Schumacher/Michael_Schumacher_0015.jpg IGNORING
	Emily_Mortimer/Emily_Mortimer_0001.jpg IGNORING
	Beth_Jones/Beth_Jones_0002.jpg IGNORING
	Narendra_Modi/Narendra_Modi_0001.jpg IGNORING
	Rob_Ramsay/Rob_Ramsay_0001.jpg IGNORING
	Lachlan_Murdoch/Lachlan_Murdoch_0001.jpg IGNORING
	Elisabeth_Schumacher/Elisabeth_Schumacher_0001.jpg IGNORING
	Jacques_Chirac/Jacques_Chirac_0006.jpg IGNORING
	Jacques_Chirac/Jacques_Chirac_0010.jpg IGNORING
	Tatiana_Panova/Tatiana_Panova_0001.jpg IGNORING
	Thor_Pedersen/Thor_Pedersen_0001.jpg IGNORING
	Jeffrey_Pfeffer/Jeffrey_Pfeffer_0001.jpg IGNORING
	Anna_Kournikova/Anna_Kournikova_0005.jpg IGNORING
	Kobe_Bryant/Kobe_Bryant_0002.jpg IGNORING
	Ricardo_Lagos/Ricardo_Lagos_0004.jpg IGNORING
	Muammar_Gaddafi/Muammar_Gaddafi_0001.jpg IGNORING
	Ed_Mekertichian/Ed_Mekertichian_0001.jpg IGNORING
	George_W_Bush/George_W_Bush_0448.jpg IGNORING
	Chan_Ho_Park/Chan_Ho_Park_0001.jpg IGNORING
	Mauricio_Pochetino/Mauricio_Pochetino_0001.jpg IGNORING
	Ken_Macha/Ken_Macha_0003.jpg IGNORING
	Jaouad_Gharib/Jaouad_Gharib_0001.jpg IGNORING
	Rudolph_Giuliani/Rudolph_Giuliani_0012.jpg IGNORING
	Rudolph_Giuliani/Rudolph_Giuliani_0007.jpg IGNORING
	Bill_OReilly/Bill_OReilly_0001.jpg IGNORING
	Lynne_Thigpen/Lynne_Thigpen_0001.jpg IGNORING
	Jessica_Lynch/Jessica_Lynch_0001.jpg IGNORING
	Annika_Sorenstam/Annika_Sorenstam_0001.jpg IGNORING
	Derrick_Rodgers/Derrick_Rodgers_0001.jpg IGNORING
	Robert_Zoellick/Robert_Zoellick_0005.jpg IGNORING
	Mark_Heller/Mark_Heller_0002.jpg IGNORING
	Rob_Moore/Rob_Moore_0001.jpg IGNORING
	Kultida_Woods/Kultida_Woods_0001.jpg IGNORING

And the output of rest of the images is as below.
```
[-0.09184992  0.08891897  0.05172213 -0.03960983 -0.09545069 -0.0170844
 -0.01534049 -0.10618818  0.13509281 -0.05730249  0.20093566 -0.0379245
 -0.24540257 -0.04531504 -0.02827422  0.12434066 -0.14732012 -0.1192862
 -0.12448209 -0.08079314  0.01167066  0.0335183   0.03045237  0.0167361
 -0.1322308  -0.29926944 -0.07019538 -0.0890242   0.05684248 -0.09198239
  0.01209295  0.05314684 -0.1847007  -0.0599186   0.02795471  0.06273955
 -0.05172921 -0.06855772  0.20853005  0.01110339 -0.1675108   0.01483111
  0.05365956  0.24574812  0.20452459  0.00596853  0.02080457 -0.07223467
  0.11354043 -0.26228943  0.04808586  0.144542    0.11229019  0.07523396
  0.08312549 -0.14238386  0.02096773  0.1370085  -0.1820543   0.06994484
  0.06932121 -0.09159457 -0.0376186  -0.04275434  0.1620959   0.08735921
 -0.09012994 -0.13499565  0.16634363 -0.14129257 -0.04414149  0.07672209
 -0.08983798 -0.15972091 -0.26815942  0.03592251  0.37520292  0.12747033
 -0.17584331  0.01636036 -0.07415752 -0.02344868  0.03634182  0.03259635
 -0.07445394 -0.04941603 -0.09306157  0.02423752  0.19745308 -0.02871043
 -0.01720607  0.21492125  0.010714   -0.00624988  0.04012334  0.04733949
 -0.1107284  -0.01356372 -0.10988513 -0.01219312  0.04726631 -0.10606344
  0.00958324  0.08684939 -0.17421511  0.15863739 -0.01775398 -0.02408041
 -0.00742908 -0.02445264 -0.07560086  0.00209789  0.17316285 -0.25108418
  0.21515246  0.17408757 -0.00231636  0.1254312   0.06280041  0.06165447
 -0.00458359 -0.00511429 -0.13308913 -0.10946756  0.02851165 -0.02758148
  0.02185735  0.03925822]
```