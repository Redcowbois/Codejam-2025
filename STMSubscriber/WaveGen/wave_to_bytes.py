import wave
import matplotlib.pyplot as plt

with wave.open('ribeiro.wav') as fd:
    params = fd.getparams()
    frames = fd.readframes(1000000) # 1 million frames max

nums = [int(c) for c in frames]
plt.plot(nums)
plt.show()

with open("ribeiro_out.txt", "w") as file:
    out = (str(nums))
    comma_counter = 0
    for c in out:
        file.write(c)
        if c == ",":
            comma_counter += 1
            if comma_counter == 10000:
                file.write("\n")
                comma_counter = 0
print(len(frames))