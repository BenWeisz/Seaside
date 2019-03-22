DEPENDENCIES = vec.h mat.h net.h

all: mnist

mnist: net.o mat.o vec.o mnist.o
	g++ -Wall -o $@ $^

flowers: net.o mat.o vec.o flowers.o
	g++ -Wall -o $@ $^

%.o: %.cpp ${DEPENDENCIES}
	g++ -Wall -c $<

clean:
	rm -f *.o flowers mnist vec mat net
