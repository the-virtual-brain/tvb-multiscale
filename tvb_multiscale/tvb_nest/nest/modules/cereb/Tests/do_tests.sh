#!/bin/bash
echo "Launching Tests for Alberto Module"
rm *.csv
rm *.gdf

#### CHECK MODELS #####

python Check_Models.py  &>TestLog.txt
if [ $? = 0 ]; then
  echo "Check_Models.py SUCCESS"
else
  echo "Check_Models.py FAIL"
fi

#### CHECK MULTITHREADING #####

mpirun -np 1 python Check_MultiThreading.py 1 &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Check_MultiThreading.py with 1 Core SUCCESS"
else
  echo "Check_MultiThreading.py with 1 Core FAIL"
fi

mpirun -np 4 python Check_MultiThreading.py 4 &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Check_MultiThreading.py with 4 Cores SUCCESS"
else
  echo "Check_MultiThreading.py with 4 Cores FAIL"
fi

python Remove_Empty.py &>>TestLog.txt

diff -u  Weights_1-* Weights_4-* > Diff_Weights.csv

if [[ -s Diff_Weights.csv ]]; then
  echo "Weights with 1 and 4 Cores are different FAIL";
else
  echo "Weights with 1 and 4 Cores are the same SUCCESS";
fi

#### CHECK STDP SINEXP #####

mpirun -np 1 python Learning_Performance_Test.py 1 &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Learning_Performance_Test.py with 1 core SUCCESS"
else
  echo "Learning_Performance_Test.py with 1 core FAIL"
fi
python Remove_Empty.py &>>TestLog.txt
mv PFPC-* PFPC1.csv

mpirun -np 4 python Learning_Performance_Test.py 4 &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Learning_Performance_Test.py with 4 Core SUCCESS"
else
  echo "Learning_Performance_Test.py with 4 Core FAIL"
fi
python Remove_Empty.py &>>TestLog.txt
cat PFPC-* > PFPC4.csv

python Compare_with_Ground_Truth1.py &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Weights with 1 and 4 Cores are the same SUCCESS"
else
  echo "Weights with 1 and 4 Cores are different FAIL"
fi

#### CHECK STDP COSEXP #####

mpirun -np 1 python Learning_Performance_Test2.py 1 &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Learning_Performance_Test2.py with 1 core SUCCESS"
else
  echo "Learning_Performance_Test2.py with 1 core FAIL"
fi
python Remove_Empty.py &>>TestLog.txt
mv MFDCN-* MFDCN1.csv

mpirun -np 4 python Learning_Performance_Test2.py 4 &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Learning_Performance_Test2.py with 4 Core SUCCESS"
else
  echo "Learning_Performance_Test2.py with 4 Core FAIL"
fi
python Remove_Empty.py &>>TestLog.txt
cat MFDCN-* > MFDCN4.csv

python Compare_with_Ground_Truth2.py &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Weights with 1 and 4 Cores are the same SUCCESS"
else
  echo "Weights with 1 and 4 Cores are different FAIL"
fi

#### CHECK iSTDP #####

mpirun -np 1 python Learning_Performance_iSTDP.py 1 &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Learning_Performance_iSTDP.py with 1 core SUCCESS"
else
  echo "Learning_Performance_iSTDP.py with 1 core FAIL"
fi
python Remove_Empty.py &>>TestLog.txt
mv iSTDP-* iSTDP1.csv

mpirun -np 4 python Learning_Performance_iSTDP.py 4 &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Learning_Performance_iSTDP.py with 4 Core SUCCESS"
else
  echo "Learning_Performance_iSTDP.py with 4 Core FAIL"
fi
python Remove_Empty.py &>>TestLog.txt
cat iSTDP-* > iSTDP4.csv
rm iSTDP-*

python Compare_with_Ground_Truth3.py &>>TestLog.txt
if [ $? = 0 ]; then
  echo "Weights with 1 and 4 Cores are the same SUCCESS"
else
  echo "Weights with 1 and 4 Cores are different FAIL"
fi

python Remove_Empty.py &>>TestLog.txt

#### CLEAN-UP THE FOLDER #####
rm *.csv *.gdf
