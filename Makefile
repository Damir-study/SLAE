main.exe: classMatrix.o main.o
	g++ classMatrix.o main.o -o main.exe

classMatrix.o: classMatrix.cpp
	g++ classMatrix.cpp -c

main.o: main.cpp 
	g++ main.cpp -c