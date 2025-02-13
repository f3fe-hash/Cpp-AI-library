C++ AI Library

This library help create simple AIs with many different activations, and train them.

| Supported activations               |
|-------------------------------------|
| tanh        (Hyperbolic tangent)    |
| relu        (Rectified linear unit) |
| sigmoid                             |
| fsigmoid    (Fast sigmoid)          |

For this test, I am using the XOR dataset (down below) and a intel i3-12th generation processor, with the example.cpp file. The lower the score the better (the score is really just the error of the network)
| Rounds        | Speed (seconds)                | Score      |
|---------------|--------------------------------|------------|
| 10            | 02.73728741                    | 0.15859183 |
| 20            | 06.48548568                    | 0.11744884 |
| 50            | 16.08952836                    | 0.15907739 |
| 100           | 34.42355963                    | 0.16801754 |

*Sometimes, the AI does worse when it is trained longer, othertimes it does worse when trained for shorter periods of time, it depends on the AI structure, settings, dataset, and how to networks are initialized.

XOR dataset:

| Input A | Input B | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

Installation
------------

To install library, just run the install.sh script or Linux, and the install.bat script for Windows, and it will compile the file and install it on to your system, so when you use it, just add -lAI as a compiler flag.

Example compilation:
g++ example.cpp -o example -lAI

To run the example after compiling:
./example
