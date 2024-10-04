{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNZxupv3U0qo4rQXXzGdwnu"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "H1OFyntn5iI_"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from keras.models import Sequential\n",
        "from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Define input sequence\n",
        "seq_in = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])\n"
      ],
      "metadata": {
        "id": "yeLCEMsg5l3z"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert characters to integers\n",
        "char_to_int = dict((c, i) for i, c in enumerate(seq_in))\n",
        "int_to_char = dict((i, c) for i, c in enumerate(seq_in))"
      ],
      "metadata": {
        "id": "LSfJyD1_5qNW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert input sequence to integers\n",
        "seq_in_int = np.array([char_to_int[char] for char in seq_in])\n"
      ],
      "metadata": {
        "id": "Ngxnv0oc5tof"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Reshape input into [samples, timesteps, features]\n",
        "n_in = len(seq_in_int)\n",
        "seq_in_int = seq_in_int.reshape((1, n_in, 1))"
      ],
      "metadata": {
        "id": "6O2sS4Iq5uRz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Prepare output sequence\n",
        "seq_out = seq_in_int[:, 1:, :]\n",
        "n_out = n_in - 1"
      ],
      "metadata": {
        "id": "kFv5zt4U5wCt"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Define model\n",
        "model = Sequential()\n",
        "model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))\n",
        "model.add(RepeatVector(n_out))\n",
        "model.add(LSTM(100, activation='relu', return_sequences=True))\n",
        "model.add(TimeDistributed(Dense(1)))\n",
        "model.compile(optimizer='adam', loss='mse')"
      ],
      "metadata": {
        "id": "q7gplt5i5yDP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Fit model\n",
        "model.fit(seq_in_int, seq_out, epochs=300, verbose=0)"
      ],
      "metadata": {
        "id": "7dt2S80p50Fj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Make prediction\n",
        "yhat = model.predict(seq_in_int, verbose=0)"
      ],
      "metadata": {
        "id": "Xe9ij74_513k"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert predictions back to characters\n",
        "predicted_chars = [int_to_char[int(round(x[0]))] for x in yhat[0]]\n",
        "print(\"Predicted next characters:\")\n",
        "print(predicted_chars)"
      ],
      "metadata": {
        "id": "z72HIjkZ54BY"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}