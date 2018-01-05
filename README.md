# beat-brain
Converting a beatbox wav file to a MIDI beat through the use of a neural net

Record a beatbox and bounce to wav file, break up wav into pieces representing each "hit" of said oral instrument, and feed a matrix of 
frequencies and their intensities (through a Fast Fourier Transform) to a neural network. The NN will hopefully recognize whether each 
"hit" is a snare, kick, or a hi-hat. Once we have that data, we can convert it to a midi file and mimic the original beat using software 
instruments.
