DEPENDENCIES = vec.h mat.h net.h

all: mnist

mnist: net.o mat.o vec.o mnist.o
	g++ -Wall -g -o $@ $^

flowers: net.o mat.o vec.o flowers.o
	g++ -Wall -g -o $@ $^

%.o: %.cpp ${DEPENDENCIES}
	g++ -Wall -g -c $<

clean:
	rm -f *.o flowers mnist vec mat net
