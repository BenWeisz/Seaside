DEPENDENCIES = vec.h mat.h net.h

all: seaside

seaside: net.o mat.o vec.o main.o
	g++ -Wall -o $@ $^

%.o: %.cpp ${DEPENDENCIES}
	g++ -Wall -c $<

clean:
	rm -f *.o main vec
