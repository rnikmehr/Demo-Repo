{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMBFPn6aKxV8QDuGoiSaUZB",
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
        "<a href=\"https://colab.research.google.com/github/rnikmehr/Demo-Repo/blob/main/Word2Vec.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "xUUf9SaNAHXo"
      },
      "outputs": [],
      "source": [
        "# Based on https://machinelearningmastery.com/develop-word-embeddings-python-gensim/\n",
        "# Import\n",
        "from gensim.models import Word2Vec"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "corpus = [\n",
        "['An', 'alpha', 'document', '.'],\n",
        "['A', 'beta', 'document', '.'],\n",
        "['Guten', 'Morgen', '!'],\n",
        "['Gamma', 'manuscript', 'is', 'old', '.'],\n",
        "['Whither', 'my', 'document', '?']\n",
        "]"
      ],
      "metadata": {
        "id": "HHHVGWxIAItV"
      },
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# train model\n",
        "# Setting min_count to 1 to ensure all words are considered, even if they appear only once.\n",
        "model = Word2Vec(corpus, min_count=1)\n",
        "# summarize the loaded model\n",
        "print(\"INFO: Model - \\n\" + str(model))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WxJMPrw4AMex",
        "outputId": "17d709b6-6d96-4a5b-b6b6-eed6cbe774b9"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO: Model - \n",
            "Word2Vec<vocab=16, vector_size=100, alpha=0.025>\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# summarize vocabulary\n",
        "# Use model.wv.key_to_index to get the vocabulary\n",
        "words = list(model.wv.key_to_index)\n",
        "print(\"INFO: Words found - \\n\" + str(words))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7SXaHpfZDVFW",
        "outputId": "719fc69c-fb0d-44b4-b77e-68d47700d3e9"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO: Words found - \n",
            "['.', 'document', '?', 'my', 'Whither', 'old', 'is', 'manuscript', 'Gamma', '!', 'Morgen', 'Guten', 'beta', 'A', 'alpha', 'An']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# access vector for one word - specified by myword\n",
        "myword = 'document'\n",
        "# Use model.wv to access word vectors\n",
        "# Checking if the word exists in the vocabulary before accessing its vector\n",
        "if myword in model.wv:\n",
        "    print(\"INFO: Model of '\" + myword + \"' - \\n\" + str(model.wv[myword]))\n",
        "else:\n",
        "    print(f\"INFO: The word '{myword}' is not present in the model's vocabulary.\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ORn-b4xFANKq",
        "outputId": "ec783221-9a53-4b83-9b38-6ba4632e37a7"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO: Model of 'document' - \n",
            "[-8.6196875e-03  3.6657380e-03  5.1898835e-03  5.7419385e-03\n",
            "  7.4669183e-03 -6.1676754e-03  1.1056137e-03  6.0472824e-03\n",
            " -2.8400505e-03 -6.1735227e-03 -4.1022300e-04 -8.3689485e-03\n",
            " -5.6000124e-03  7.1045388e-03  3.3525396e-03  7.2256695e-03\n",
            "  6.8002474e-03  7.5307419e-03 -3.7891543e-03 -5.6180597e-04\n",
            "  2.3483764e-03 -4.5190323e-03  8.3887316e-03 -9.8581640e-03\n",
            "  6.7646410e-03  2.9144168e-03 -4.9328315e-03  4.3981876e-03\n",
            " -1.7395747e-03  6.7113843e-03  9.9648498e-03 -4.3624435e-03\n",
            " -5.9933780e-04 -5.6956373e-03  3.8508223e-03  2.7866268e-03\n",
            "  6.8910765e-03  6.1010956e-03  9.5384968e-03  9.2734173e-03\n",
            "  7.8980681e-03 -6.9895042e-03 -9.1558648e-03 -3.5575271e-04\n",
            " -3.0998408e-03  7.8943167e-03  5.9385742e-03 -1.5456629e-03\n",
            "  1.5109634e-03  1.7900408e-03  7.8175711e-03 -9.5101865e-03\n",
            " -2.0553112e-04  3.4691966e-03 -9.3897223e-04  8.3817719e-03\n",
            "  9.0107834e-03  6.5365066e-03 -7.1162102e-04  7.7104042e-03\n",
            " -8.5343346e-03  3.2071066e-03 -4.6379971e-03 -5.0889552e-03\n",
            "  3.5896183e-03  5.3703394e-03  7.7695143e-03 -5.7665063e-03\n",
            "  7.4333609e-03  6.6254963e-03 -3.7098003e-03 -8.7456414e-03\n",
            "  5.4374672e-03  6.5097557e-03 -7.8755023e-04 -6.7098560e-03\n",
            " -7.0859254e-03 -2.4970602e-03  5.1432536e-03 -3.6652375e-03\n",
            " -9.3700597e-03  3.8267397e-03  4.8844791e-03 -6.4285635e-03\n",
            "  1.2085581e-03 -2.0748770e-03  2.4403334e-05 -9.8835090e-03\n",
            "  2.6920044e-03 -4.7501065e-03  1.0876465e-03 -1.5762246e-03\n",
            "  2.1966731e-03 -7.8815762e-03 -2.7171839e-03  2.6631986e-03\n",
            "  5.3466819e-03 -2.3915148e-03 -9.5100943e-03  4.5058788e-03]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "\n",
        "# ... your existing code ...\n",
        "\n",
        "# Create the 'data' directory if it doesn't exist\n",
        "os.makedirs('../data', exist_ok=True)\n",
        "\n",
        "# save model\n",
        "model.save('../data/model.bin')\n",
        "model.wv.save_word2vec_format('../data/model.txt', binary=False)"
      ],
      "metadata": {
        "id": "zzDJcCFKAQ1a"
      },
      "execution_count": 31,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# load model\n",
        "new_model = Word2Vec.load('../data/model.bin')\n",
        "print(\"INFO: Reloaded Model - \\n\" + str(new_model))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qGR9o5L6ASt3",
        "outputId": "786c2e26-901f-4277-8c5e-aaca46355750"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO: Reloaded Model - \n",
            "Word2Vec<vocab=16, vector_size=100, alpha=0.025>\n"
          ]
        }
      ]
    }
  ]
}