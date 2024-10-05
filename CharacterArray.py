{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNk4wPnh5tKs/4xzKAFXIK6",
      "include_colab_link": true
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
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rnikmehr/Demo-Repo/blob/main/CharacterArray.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "H1OFyntn5iI_"
      },
      "outputs": [],
      "source": [
        "from keras.models import Sequential\n",
        "from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input"
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
      "execution_count": 11,
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
      "execution_count": 12,
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
      "execution_count": 13,
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
      "execution_count": 14,
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
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Define model\n",
        "model = Sequential([\n",
        "    Input(shape=(n_in, 1)),\n",
        "    LSTM(100, activation='relu'),\n",
        "    RepeatVector(n_out),\n",
        "    LSTM(100, activation='relu', return_sequences=True),\n",
        "    TimeDistributed(Dense(1))\n",
        "])\n",
        "\n",
        "model.compile(optimizer='adam', loss='mse')"
      ],
      "metadata": {
        "id": "q7gplt5i5yDP"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Fit model\n",
        "model.fit(seq_in_int, seq_out, epochs=300, verbose=0)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7dt2S80p50Fj",
        "outputId": "ee86e91e-d030-45e0-f9a1-b39a3d42086a"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.history.History at 0x7941e76d5f30>"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ]
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
      "execution_count": 18,
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
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "z72HIjkZ54BY",
        "outputId": "2999ef55-52f5-4916-cf16-1e0ecae415ad"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Predicted next characters:\n",
            "['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']\n"
          ]
        }
      ]
    }
  ]
}