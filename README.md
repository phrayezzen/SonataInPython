# SonataInPython
Algorithmically generated musical literature using neural networks

Inspiration
Mozart may write his sonatas in C, but my C is just not as good. So, I wrote mine in Python. I've always built webapps and I just finished an ML class so I wanted to try it out. Plus, now we can see if art can be scientifically generated?

What it does
Sonata in Python reads in a series of MIDI files, mostly piano sonatas by Mozart or Beethoven. This acts as the training set for the neural network. The input nodes for the network take the duration and frequency of the preceding 5 notes as well as the current note, the preceding 3 chord progressions, and the corresponding note on the other clef. The output nodes produce the duration and frequency of the next note in the sequence, which is fed back as an input for the subsequent note.

How I built it
There is one neural network for the treble clef (right hand) and one for the bass clef (left hand). This is because the higher right hand notes often play a unique melody, while the lower left hand notes are often repeated, slower chords. I wanted to make sure these two very separate styles of music didn't confuse the neural network.

Challenges I ran into
The biggest challenge is that feeding outputs back into the neural network will cause the network to converge. In the beginning, my compositions often converged to a single, repeating note. This produced extremely boring music. I fixed this by introducing mutations to the network. First I tried to sample a Gaussian distribution based on the mean and standard deviation of the error vectors. Then, I switched to a small percentage of changing the note by a perfect fifth or a perfect octave, both of which do not affect the flow of the music much. Furthermore, both the treble and bass clefs converged to the same note. I fixed this problem by splitting the the network in two, as mentioned earlier. Finally, neural networks do not give very exact predictions. Unfortunately, in music, even the slightest deviation will cause the music to sound out of tune. I often had quarter tones and random durations of notes. I fixed this by rounding the pitch and duration, which significantly increased the quality of the music.

Accomplishments that I'm proud of
I'm proud of having done this project alone. I've been to a lot of hackathons in past years, but always under the tutelage of upperclassmen. Now that they've graduated, I'm sort of wondering on my own. I'm glad I was able to accomplish this project without others directing me. I'm also proud of how far I came with this project. I compared my first generated sonata to my most recent one and I was really impressed with how good the latter one sounded. Perhaps not yet at the level of Mozart, but it's certainly come a long way.

What I learned
Music is hard! 36 hours is not enough to create an algorithm that can generate music that has been endeared for centuries. I have a much greater appreciation (but also greater frustration...). Python is incredible. Technically, I already knew this, but to be able to produce this quality of a music generating, machine learning algorithm in under 250 lines is incredible. Yes, I only wrote 250 [very well thought out and tested via trial and error!] lines. Python is life. I was also introduced to pybrain and music21, Python libraries used for neural networks and analyzing music, respectively. Both of these are very well made and I hope I have an opportunity to explore more of their features in future projects. Also, I had to learn a lot of stat. It was pretty uncomfortable, and I'm not a fan...

What's next for Sonata in Python
Obviously, more fine tuning of parameters in order to produce even more beautiful music. While I did consider chords in the neural network, harmony is probably the most important missing feature. Harmony analysis is a 36 hour project on its own, and would greatly improve the project. In addition, I'm hoping to expand to concertos and contemporary music. Sonatas are fairly easier because they are generally piano only, with only a left and right hand. Opening to more types of music would be very fun.

PS. Unfortunately, I have no pictures! There's not much to show about a neural network... However, 10 MIDI files of varying quality can be found on the Github.

Built With
Python, PyBrain, Music21
