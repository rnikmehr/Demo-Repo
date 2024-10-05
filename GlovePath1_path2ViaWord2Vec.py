{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOYQS6gQzxXxD5rpiPPtIy+",
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
        "<a href=\"https://colab.research.google.com/github/rnikmehr/Demo-Repo/blob/main/GlovePath1_path2ViaWord2Vec.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Q2gwoDN8J1a0"
      },
      "outputs": [],
      "source": [
        "## Based on\n",
        "# - [1] https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db\n",
        "# - [2] https://machinelearningmastery.com/develop-word-embeddings-python-gensim/\n",
        "# - [3] Glove site: https://nlp.stanford.edu/projects/glove/"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Install as necessary\n",
        "# !pip install numpy\n",
        "# !pip install scipy\n",
        "# !pip install matplotlib\n",
        "# !pip install sklearn"
      ],
      "metadata": {
        "id": "5RDpX8ttJ7MM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Necessary imports\n",
        "import numpy as np\n",
        "from sklearn.manifold import TSNE"
      ],
      "metadata": {
        "id": "31-ZCSFpJ_Aq"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Copy data from: https://nlp.stanford.edu/data/glove.6B.zip\n",
        "# Unzip and use file glove.6B.50d.txt; set the data path accordingly\n",
        "# Replace 'path/to/glove.6B.50d.txt' with the actual path to your file\n",
        "glove_input_file = '/content/glove.6B.50d.txt'"
      ],
      "metadata": {
        "id": "xN9bswbKKFz9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Path 1\n",
        "embeddings_dict = {}\n",
        "with open(glove_input_file, 'r') as f:\n",
        "    for line in f:\n",
        "        values = line.split()\n",
        "        word = values[0]\n",
        "        # Wrap vector creation in a try-except to handle invalid lines\n",
        "        try:\n",
        "            vector = np.asarray(values[1:], \"float32\")\n",
        "            # Check if the vector has the correct dimension (50 in this case)\n",
        "            if vector.shape[0] == 50:\n",
        "                embeddings_dict[word] = vector\n",
        "            else:\n",
        "                print(f\"Skipping word '{word}' due to incorrect vector dimension: {vector.shape}\")\n",
        "        except ValueError:\n",
        "            print(f\"Skipping word '{word}' due to invalid vector values\")"
      ],
      "metadata": {
        "id": "bE8vEq9oKGyL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# A function defined for similarity\n",
        "# - See description of euclidean use in [1]\n",
        "def find_closest_embeddings(embedding):\n",
        "    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))\n"
      ],
      "metadata": {
        "id": "16gznf6aUBhT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Find closest word\n",
        "find_closest_embeddings(embeddings_dict[\"king\"])[:5]"
      ],
      "metadata": {
        "id": "ReaJ86_VKJjO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0cc1914d-4e44-46bb-8cb9-e3b6c4673dce"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['king', 'prince', 'queen', 'uncle', 'ii']"
            ]
          },
          "metadata": {},
          "execution_count": 80
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Vector operation\n",
        "print(find_closest_embeddings(\n",
        "    embeddings_dict[\"twig\"] - embeddings_dict[\"branch\"] + embeddings_dict[\"hand\"]\n",
        ")[:5])"
      ],
      "metadata": {
        "id": "8ZQSMlBAKNKj",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "56ff1572-08c3-4ac1-d39d-5729367ac32d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['fingernails', 'toenails', 'stringy', 'peeling', 'shove']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# For visualizing\n",
        "tsne = TSNE(n_components=2, random_state=0)"
      ],
      "metadata": {
        "id": "fdE3E8gTKREf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Organizing data structures\n",
        "words = list(embeddings_dict.keys())\n",
        "vectors = [embeddings_dict[word] for word in words]"
      ],
      "metadata": {
        "id": "NYYND5OTKRvI"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert the list of arrays to a 2D NumPy array\n",
        "vectors_array = np.array(vectors)  # Convert 'vectors' to a NumPy array"
      ],
      "metadata": {
        "id": "FMLkK3m0KVFp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "Y = tsne.fit_transform(vectors_array[:50]) # Use the NumPy array for fit_transform\n"
      ],
      "metadata": {
        "id": "bkAxUptOX72d"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(Y[:, 0], Y[:, 1])\n",
        "\n",
        "for label, x, y in zip(words, Y[:, 0], Y[:, 1]):\n",
        "    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords=\"offset points\")\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 430
        },
        "id": "InprA68NYEns",
        "outputId": "933b50e8-80f3-48f7-d3a2-8e1cd15c8c55"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAi8AAAGdCAYAAADaPpOnAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAB0pUlEQVR4nO3deXxM5/7A8c8kkkzWiZBIQkgQSwRBGmJNlSuKW3RBqZ1SiqKW2xahrWpLLb2l1Va0KG4tpdWopUlRBBFrUCFSTMSaSCLrzO+P/DI1skjIZCbJ9/16zetmzjznzPfM1TnfeZ7nfB+FVqvVIoQQQghRTpgZOwAhhBBCiJKQ5EUIIYQQ5YokL0IIIYQoVyR5EUIIIUS5IsmLEEIIIcoVSV6EEEIIUa5I8iKEEEKIckWSFyGEEEKUK1WMHUBp02g0XL9+HXt7exQKhbHDEUIIIUQxaLVa7t+/j7u7O2ZmRfetVLjk5fr163h4eBg7DCGEEEI8gb///ptatWoV2abCJS/29vZA7sk7ODgYORohhBBCFEdycjIeHh6663hRKlzykjdU5ODgIMmLEEIIUc4UZ8qHTNgVQgghRLkiyYsQQgghyhVJXoQQQghRrkjyIoQodaGhoTg6Oho7DCFEBSXJixBCCCHKFUlehBBCCFGuSPIihCiWn3/+GUdHR3JycgCIjo5GoVAwY8YMXZuRI0cyaNAg3fOdO3fSuHFj7OzsCA4ORq1W617TaDTMnTuXWrVqYWVlhZ+fH2FhYWV3QkKIckuSFyFEsXTo0IH79+9z/PhxACIiIqhevTrh4eG6NhEREQQFBQGQlpbGp59+yvfff88ff/xBfHw8U6dO1bVdsmQJc+fOpWnTppw8eZJu3brx73//m7/++kvXRqFQsHXr1rI4PSFEOSLJixCiUDkaLQdjb/NT9DXO3srGz89Pl6yEh4fz1ltvcfz4cVJSUrh27RoXL16kU6dOAGRlZbFixQr8/f1p2bIl48ePZ8+ePbpjf/rpp3h4eNCwYUMaNmzIggUL8PPzY/Hixbo2arWa7t27l+UpCyHKAUlehBAFCjutpv2CvQxYeYiJ66MZsPIQ1629+N/2nWi1Wvbt20ffvn1p3Lgx+/fvJyIiAnd3d7y9vQGwsbGhXr16uuO5ubmRmJgI5JYBv379OiqVSu8927VrR0xMjO65q6srVlZWZXC2QojyRJIXIUQ+YafVjF0ThTopXW+7poYPRw4f5ItNu7GwsKBRo0YEBQURHh5ORESErtcFwMLCQm9fhUKBVqvN914ajYZp06bh5OTEV199xeXLl/X2yRs2yszMZPz48bi5uaFUKqlTpw7z588vxbMWQpQXkrwIIfTkaLSEbD9L/jQDLD2aoM18wJwPP6Fjx9xEJS95CQ8P1813eRwHBwfc3d1JSkpi9erV2NracvjwYWrUqEFcXBy7du3Kt8/SpUvZtm0bGzdu5Pz586xduxZPT88nP1EhRLklyYsQQk/k5Tv5elzymCvtsHD25Fb0Hur4+gPQsWNHoqKiuHDhgl7Py+O8/fbb/P3337i5udG/f3+++eYbrl69StOmTfXmxuSJj4/H29ub9u3bU6dOHdq3b8+AAQOe7CSFEOWaJC9CCD2J9wtOXPIoPXxBq6GO7zMAODk54ePjg6urKw0bNnzs8fMmAdfp+CLOru66hCUsLIxt27bh5eWlmxvzsKFDhxIdHU3Dhg2ZMGECv/3225OdoBCi3JPkRQihx8VeWeTrTl1GU2f6z7Rs5qvbFh0drVfDZejQody7d09vv969e/Prqeu6ScBvbTzJXTMV9s27sS3qCtHR0QQHB6NQKNBoNPnet2XLlly+fJl58+bx4MEDXnnlFV566aWnO1khRLkkyYsQQk+AlxNuKiWKQl5XAG4qJQFeTiU6bmGTgFMzshm7Joqw0+pC9vyHg4MD/fr1Y+XKlWzYsIFNmzZx586dEsUhhCj/JHkRQugxN1Mwu5cPQL4EJu/57F4+mJsVlt7kV9Qk4Dwh28+Soym8xaJFi/jhhx84d+4cFy5c4H//+x+urq6yAKQQlZAkL0KIfIJ93Vg+qCWuKv0hJFeVkuWDWhLs61ai4xU1CRhAC6iT0om8XHgvir29PR9//DH+/v4888wzxMXFsWPHDszM5GtMiMpGoS2o8EI5lpycjEqlIikpCQcHB2OHI0S5lqPREnn5Don303Gxzx0qKkmPS56foq8xcX30Y9st6e/HC341i3XMoKCgfBV5hRDlV0mu31XKKCYhRDlkbqYgsF61pz7O4yYBl7SdEKJyk/5WIYTBGWoSsBCicpLkRQhhcIaYBAz6Swu4uroyZ84c3Wv37t1j5MiRODs74+DgQOfOnTlx4sSTn4QQwmRI8iKEKBOlPQkY0Fta4OOPP2bu3Lm6pQVefvllEhMT+fXXXzl27BgtW7bkueeek1urhagADDphd/78+WzevJlz585hbW1N27ZtWbBgwWOrcP7vf//jvffeIy4uDm9vbxYsWMDzzz9frPeUCbtCmLanmQT88L4hr7+CjaUZ+/ft070eEBBA586d6dmzJz169CAxMVFvVer69eszbdo0Ro8eXernJYR4OiYzYTciIoJx48bxzDPPkJ2dzX/+8x/+9a9/cfbsWWxtbQvc588//2TAgAHMnz+fnj17sm7dOnr37k1UVBS+vr4F7iOEKD+edBJw2Gk1IdvP6m65TlAn4+hel7DTal2vjZubG4mJiZw4cYKUlBSqVdN/nwcPHhAbG/v0JyGEMKoyvVX65s2buLi4EBERQceOHQts069fP1JTU/n5559129q0aYOfnx8rVqx47HtIz4sQFU9edd6Hv6wS1s3A0qUu1bqM1g079e7dG0dHRxo3bsyyZcsIDw/PdyxHR0eqV69eZrELIYrHZHpeHpWUlATkLuRWmIMHDzJ58mS9bd26dWPr1q2GDE0IYaKKW523q4+r7nnLli1JSEigSpUqeHp6GjxGIUTZKrMJuxqNhkmTJtGuXbsih38SEhKoUaOG3rYaNWqQkJBQYPuMjAySk5P1HkKIiuNJqvN26dKFwMBAevfuzW+//UZcXBx//vkn77zzDkePHi2DqIUQhlRmycu4ceM4ffo069evL9Xjzp8/H5VKpXt4eHiU6vGFEMaVeL/wxKWwdgqFgh07dtCxY0eGDRtGgwYN6N+/P1euXMn340gIUf6UybDR+PHj+fnnn/njjz+oVatWkW1dXV25ceOG3rYbN27g6upaYPuZM2fqDTMlJydLAiNEBVJY1V3XVz/K1+7h4WV7e3uWLl3K0qVLDRmeEMIIDNrzotVqGT9+PFu2bGHv3r14eXk9dp/AwED27Nmjt23Xrl0EBgYW2N7KygoHBwe9hxCi4pDqvEKIRxk0eRk3bhxr1qxh3bp12Nvbk5CQQEJCAg8ePNC1GTx4MDNnztQ9nzhxImFhYSxcuJBz584xZ84cjh49yvjx4w0ZqhDCRBmqOq8QovwyaPKyfPlykpKSCAoKws3NTffYsGGDrk18fDxqtVr3vG3btqxbt46vvvqK5s2b8+OPP7J161ap8SJEJWaI6rxCiPKrTOu8lAWp8yJExfU01XmFEKbNZOu8CCHE03jS6rxCiIpFFmYUQgghRLkiyYsQQgghyhVJXoQQQnDu3DnatGmDUqnEz8/P2OEIUSRJXoQQQjB79mxsbW05f/48e/bsITQ0FEdHR2OHJUSBZMKuEEIIYmNj6dGjB3Xq1CnV4+bk5KBQKDAzk9/KovTIvyYhhKgEwsLCaN++PY6OjlSrVo2ePXsSGxsL5K4FdezYMebOnYtCoSAoKIhhw4aRlJSEQqFAoVAwZ84cIHcx3KlTp1KzZk1sbW1p3bo14eHhuvfJ67HZtm0bPj4+WFlZER8fb4QzFhWZJC9CCFEJpKamMnnyZI4ePcqePXswMzOjT58+aDQa1Go1TZo0YcqUKajVarZt28bixYtxcHBArVajVquZOnUqkLtW3cGDB1m/fj0nT57k5ZdfJjg4mL/++kv3XmlpaSxYsICvv/6aM2fO4OLiYqzTFhWUDBsJIUQF9GhBv959+uoV9Pv2229xdnbm7Nmz+Pr6UqVKFezs7HSL4KpUKhQKhd6iuPHx8axatYr4+Hjc3d0BmDp1KmFhYaxatYoPP/wQgKysLL744guaN29ehmcsKhPpeRFCiGL6+eefcXR0JCcnB4Do6GgUCgUzZszQtRk5ciSDBg3i9u3bDBgwgJo1a2JjY0PTpk354Ycf9I73448/0rRpU6ytralWrRpdunShQ4cOTJo06aniDDutpv2CvQxYeYiJ66PpNeF9LK1tCXq+N3Xr1sXBwQFPT0+AEg3pnDp1ipycHBo0aICdnZ3uERERoRuCArC0tKRZs2ZPdQ5CFEV6XoQQopg6dOjA/fv3OX78OP7+/kRERFC9enW9OR8RERFMnz6d9PR0WrVqxfTp03FwcOCXX37htddeo169egQEBKBWqxkwYAAff/wxffr04f79++zbty9fglNSYafVjF0TxaPrvmiy0jkcE8+sdz6id7umaDQafH19yczMLPaxU1JSMDc359ixY5ibm+u9Zmdnp/vb2toahUKWbRCGIz0vQghRTCqVCj8/P12yEh4ezltvvcXx48dJSUnh2rVrXLx4kU6dOlGzZk2mTp2Kn58fdevW5c033yQ4OJiNGzcCoFaryc7Opm/fvnh6etK0aVPeeOONfElBSeRotIRsP5s/ccl8AFotjm37se1mNRo0bMTdu3eLPJalpaWuhylPixYtyMnJITExkfr16+s9Hh5eEsLQJHkRQojHyNFoORh7m5+ir+HdPIDffw9Hq9Wyb98++vbtS+PGjdm/fz8RERG4u7vj7e1NTk4O8+bNo2nTpjg5OWFnZ8fOnTt1wzTNmzfnueeeo2nTprz88susXLlSl1BoNBqmTZuGk5MTrq6uujt9ABYtWkTTpk2xtbXFw8ODN954g5SUFAAiL99BnZROyqndXP1iGPELXyRx8/tos3N7V+6f2El83CW+WLeVyZMnF3nOnp6epKSksGfPHm7dukVaWhoNGjRg4MCBDB48mM2bN3P58mUiIyOZP38+v/zyiwE+eSEKJsmLEEIU4dH5I3uTnNm5N5wvNu3GwsKCRo0aERQURHh4OBEREXTq1AmATz75hCVLljB9+nR+//13oqOj6datm26YxtzcnF27dvHrr7/i4+PDsmXLaNiwIenp6axevRpbW1sOHz7Mxx9/zNy5c9m1axcAZmZmLF26lDNnzrB69Wr27t3LtGnTAEi8n07G9fPc/nUp9q164DZsKcrazUg+uBEslGQmXOT6N+NYNPcdPvnkkyLPu23btowZM4Z+/frh7OzMxx9/DMCqVasYPHgwU6ZMoWHDhvTu3ZsjR45Qu3ZtQ/1fIEQ+Cq1W+2gPY7lWkiW1hRCiKAXNH8lJT+Hq0lexbRJEq1p2RPz6E1u3buWjjz7i7t27TJkyhdGjR9OrVy9cXFz45ptvgNzelEaNGuHj48PWrVv13idHo+XgxZv07tAcpVKJZ+1a7N+3T/d6QEAAnTt35qOPPsoX448//siYMWO4desWB2Nv06XXi2gzUnF5eY6uzc2fFvDgchS1J20A4IdRbWR1bmFySnL9lgm7QghRgMLmj5gr7bBw9iT1TDh/13uTHI2Wjh078sorr5CVlaXrefH29ubHH3/kzz//pGrVqixatIgbN27g4+MDwOHDh9mzZw929VoSGnWHv8+f4vbtW1hUrUl2lhNhp9UE+7oB4ObmRmJiIgC7d+9m/vz5nDt3juTkZLKzs0lPTyctLY0ALye4dxWrem30Yraq2YgHl6NQAK4qZW47IcoxGTYSQogC5M0fKYjSwxe0GjKcGxF5+Q5OTk74+Pjg6upKw4YNAXj33Xdp2bIl3bp1IygoCFdXV3r37q07hoODA5t37OKt4f05vnAo9/Z9T9VnR2BmbU9aNoxdE0XYaTWQWwFXo9EQFxdHz549adasGZs2beLYsWP897//BSAzMxNzMwWuKmXuPoWc1+xePnr1XoQoj6TnRQghCpB4v+DEBcCpy2icuozWaxcdHa3fxskp3/DQwxo0bIRFj3fxaK//PmnnD+j+Dtl+lq4+/9zFc+zYMTQaDQsXLtStFZR391Ke1i2acfFqAuYqpS75yrh+HjMFLB/UUtebI0xTZmYmlpaWxg7D5EnPixBCFMDFXlmq7R5VVM8OgBZQJ6UTefmOblv9+vXJyspi2bJlXLp0ie+//54VK1bo7TdhwgSOHfidPlWi+LirM//iOObXTmBnVUUSl1JQkkKFAPv376dDhw5YW1vj4eHBhAkTSE1N1bX19PRk3rx5DB48GAcHB0aPHl2s/So7SV6EEKIAAV5OuKmUhQ6/KAC3p5g/UlTPTmHtmjdvzqJFi1iwYAG+vr6sXbuW+fPn67Vv06YNK1euZNnSpQztFUTC2Uhmz3rviWIU+T1cqBAotFBhUFAQsbGxBAcH8+KLL3Ly5Ek2bNjA/v37GT9+vN4xP/30U5o3b87x48d57733ir1fZSZ3GwkhRCHy7jYC9Cbu5iU0TzMMczD2NgNWHnpsO7kzyDQ8vFbU9EHPM3zIIKa9/TZ9+vThmWeeISQkhNu3b5OUlEStWrW4cOECCxYswNzcnC+//FJ3nP3799OpUydSU1Nz7yzz9KRFixZs2bJF12bkyJGP3a8ikruNhBCiFAT7urF8UEtCtp/VG+JxVSmZ3cvnqYZh8np2EpLS893RBMidQSYk7LRa79/AHWsv5n+ziabBA9m3bx/z589n48aN7N+/nzt37ugKFZ44cYKTJ0+ydu1a3bG0Wi0ajYbLly/TuHFjAPz9/fXer7j7VWaSvAghRBGCfd3o6uOqt0JzgJfTU9+xY26mYHYvH8auyb2FuaCeHbkzyPgKqvWjrN2MW7/sZsSiTWgU5nqFCu/evau7XT4lJYXXX3+dCRMm5Dvuw0X9bG1t9V4r7n6VmSQvQgjxGOZmCoMM3RiyZ0c8vcJq/Vh5NEGb+YDko1tRuvmQo9ESFBSkV6gQoGXLlpw9e5b69euX6H2fdL/KRJIXIYQwIkP17IinV9gdYQ8XKrTqOobIy3cKLFQ4ffp02rRpw/jx4xk5ciS2tracPXuWXbt28fnnnxf6vk+6X2UiyYsQQhiZoXp2xNMp6o4wpYcvWYmXUNZuSuL9dALr1cTHx4cbN27oChU2a9aMiIgI3nnnHTp06IBWq6VevXr069evyPd90v0qE7nbSAghhCiA3BFWtkpy/ZY6L0IIIUQBDF3rRzw5SV6EEEKIAuTdEQb514qSO8KMS5IXUa4oFIoi14uJi4tDoVDkW2dGCCGeRN4dYXkLXuZxVSllrSgjkgm7olxRq9VUrVrV2GEIISoRuSPM9EjyIsoVV1fXxzcSQohSJneEmRYZNhJl7scff6Rp06ZYW1tTrVo1unTpQmpqKkeOHKFr165Ur14dlUpFp06diIqK0tv30WGjyMhIWrRogVKpxN/fX7dYmhBCiIpLkhdRptRqNQMGDGD48OHExMQQHh5O37590Wq13L9/nyFDhrB//34OHTqEt7c3zz//PPfv3y/wWCkpKfTs2RMfHx+OHTvGnDlzmDp1ahmfkRBCiLJm0GGjP/74g08++YRjx46hVqvZsmULvXv3LrR9eHg4zz77bL7tarVahgvKubwVWQ9GniA7O5sXevfB09MTgKZNmwLQuXNnvX2++uorHB0diYiIoGfPnvmOuW7dOjQaDd988w1KpZImTZpw9epVxo4da/DzEUIIYTwGTV5SU1Np3rw5w4cPp2/fvsXe7/z583oFalxcXAwRnigjD6/IqtXkoKzTnAaNmxDYsTODX/43L730ElWrVuXGjRu8++67hIeHk5iYSE5ODmlpacTHxxd43JiYGJo1a6a3PHxgYGBZnZYQQggjMWjy0r17d7p3717i/VxcXHB0dCz9gESZe3RFVoWZOS793ifzWgynLh/nw08+45133uHw4cOMHTuW27dvs2TJEurUqYOVlRWBgYFkZmYa9RyEEEKYFpOc8+Ln54ebmxtdu3blwIEDRbbNyMggOTlZ7yFMQ2ErsioUCqxq+VC1w0BqDFmMpaUlW7Zs4cCBA0yYMIHnn3+eJk2aYGVlxa1btwo9fuPGjTl58iTp6f+sP3Lo0ONLeQshhCjfTCp5cXNzY8WKFWzatIlNmzbh4eFBUFBQvjtOHjZ//nxUKpXu4eHhUYYRi6IUtCJrxvXzJB3cSIb6L7KSE4k98juJiTdp3Lgx3t7efP/998TExHD48GEGDhyItbV1ocd/9dVXUSgUjBo1irNnz7Jjxw4+/fRTQ5+WEEIIIzOpOi8NGzbUrcYJ0LZtW2JjY/nss8/4/vvvC9xn5syZTJ48Wfc8OTlZEhgTUdCKrGaWNqT/fZrkoz+hyUijisqFYZPfo3v37ri6ujJ69GhatmyJh4cHH374YZF3D9nZ2bF9+3bGjBlDixYt8PHxYcGCBbz44ouGPC0hhBBGZlLJS0ECAgLYv39/oa9bWVlhZWVVhhGJ4nKxV+bbZlHdgxqvzNXbNmxUGwBatGjBkSNH9F576aWX9J4/ugh6mzZt8i0FUMEWShdCCPEIkxo2Kkh0dDRubrJ2RHkkK7IKIYQwBIP2vKSkpHDx4kXd88uXLxMdHY2TkxO1a9dm5syZXLt2je+++w6AxYsX4+XlRZMmTUhPT+frr79m7969/Pbbb4YMUxhI3oqsY9dEoQC9ibuyIqsQQognZdDk5ejRo3pF5/LmpgwZMoTQ0FDUarVeDY/MzEymTJnCtWvXsLGxoVmzZuzevbvAwnWifMhbkTWvzkseV5WS2b18ZEVWIYQQJabQVrAJAsnJyahUKpKSkvQK3QnjyquwKyuyCiGEKEhJrt8mP2FXVAyyIqsQQojSYvITdoUQQgghHibJixBCCCHKFUlehBBCCFGuSPIihBAmKCgoiEmTJhk7DCFMkiQvQgghhChXJHkRQgghRLkiyYsQQpio7Oxsxo8fj0qlonr16rz33ntotVrmzp2Lr69vvvZ+fn689957RohUiLIlyYsQQpio1atXU6VKFSIjI1myZAmLFi3i66+/Zvjw4cTExOgtZHr8+HFOnjzJsGHDjBixEGVDitQJIYSJ8vDw4LPPPkOhUNCwYUNOnTrFZ599xqhRo+jWrRurVq3imWeeAWDVqlV06tSJunXrGjlqIQxPel6EEMIE5Gi0HIy9zU/R1zgYexst0KZNGxSKf5bRCAwM5K+//iInJ4dRo0bxww8/kJ6eTmZmJuvWrWP48OHGOwEhypD0vAghhJGFnVbnW7z0TvxdrKqmFbpPr169sLKyYsuWLVhaWpKVlcVLL71UFuEKYXSSvAghhBGFnVYzdk0Uj66Qm5mtIXzfQcJOq3Wrrx86dAhvb2/Mzc0BGDJkCKtWrcLS0pL+/ftjbW1dxtELYRySvAghhJHkaLSEbD+bL3HJk33/JsPGvMmele9zIvo4y5YtY+HChbrXR44cSePGjQE4cOBAGUQshGmQ5EUIIYwk8vIdvaGiR9k26UxKahoBAQFYWlRh4sSJjB49Wve6t7c3bdu25c6dO7Ru3bosQhbCJEjyIoQQRpJ4v/DExfXVj3R/L1m1khf8auZro9VquX79Om+88YZB4hPCVEnyIoQQRuJir3zidjdv3mT9+vUkJCRIbRdR6UjyIoQQRhLg5YSbSklCUnqB814UgKtKSYCXU77XXFxcqF69Ol999RVVq1Y1eKxCmBJJXoQQwkjMzRTM7uXD2DVRKEAvgcmr7jK7lw/mZop8+2q1hU3zFaLikyJ1QghhRMG+biwf1BJXlf7QkKtKyfJBLXW3SQsh/iE9L0JUUkOHDsXT05M5c+YYO5RKL9jXja4+rkRevkPi/XRc7HOHigrqcRFCSPIihBAmwdxMQWC9asYOQ4hyQYaNhBB88cUXeHt7o1QqqVGjhpSZF0KYNOl5EaKSO3r0KBMmTOD777/XFTzbt2+fscMSQohCKbQVbMp6cnIyKpWKpKQkHBwcjB2OECZv8+bNDBs2jKtXr2Jvb2/scIQQlVRJrt8ybCREJZGj0XIw9jY/RV/jYOxtcjS5v1u6du1KnTp1qFu3Lq+99hpr164lLa3w1YyFEMLYpOdFiEog7LSakO1n9dbRcVMpmd3Lh2BfN7KzswkPD+e3335j06ZNmJmZceTIERwdHY0XtBCiUinJ9VuSFyEquLDTasauicpXwTXvJtxHa4mkpqbi6OjIhg0b6Nu3b5nFKYSo3GTYSAgB5A4VhWw/W2Dp+bxtkz75lsVLlhAdHc2VK1f47rvv0Gg0NGzYsCxDFUIUIS4uDoVCQXR0tLFDMQlyt5EQFVjk5Tt6Q0WP0gL3ciz4bt1G5oaEkJ6ejre3Nz/88ANNmjQpu0CFEKIEpOdFiAos8X7hiUseZa0mzP5yI3fu3CEtLY0TJ07wyiuvlEF0QoiHhYWF0b59exwdHalWrRo9e/YkNjYWAC8vLwBatGiBQqEgKCjIiJEanyQvQlRgLvbKxzcqQTshhOGkpqYyefJkjh49yp49ezAzM6NPnz5oNBoiIyMB2L17N2q1ms2bNxs5WuOSYSMhKrAALyfcVEoSktILnPeiIHcBwAAvp7IOTQjxiBdffFHv+bfffouzszNnz57F2dkZgGrVquHq6mqM8EyKQXte/vjjD3r16oW7uzsKhYKtW7c+dp/w8HBatmyJlZUV9evXJzQ01JAhClGhmZspmN3LB/jn7qI8ec9n9/KRBQCFMIJHay+dO3+BAQMGULduXRwcHPD09AQgPj7euIGaIIP2vKSmptK8eXOGDx9erFsuL1++TI8ePRgzZgxr165lz549jBw5Ejc3N7p162bIUIWosIJ93Vg+qGW+Oi+uD9V5EUKUrYJqL934Ziw+DeqycuVK3N3d0Wg0+Pr6kpmZacRITZNBk5fu3bvTvXv3YrdfsWIFXl5eLFy4EIDGjRuzf/9+PvvsM0lehHgKwb5udPVxJfLyHRLvp+NinztUJD0uQpS9gmov5TxIJv3W31z/1ziyavjQuLEb+/fv171uaWmZ2y4np4yjNU0mNefl4MGDdOnSRW9bt27dmDRpUqH7ZGRkkJGRoXuenJxsqPCEKNfMzRQE1qtm7DCEqNQKq71kprTDzNqB+yd2MjPUDfPgmrzzn5m6111cXLC2tiYsLIxatWqhVCpRqVRlG7wJMam7jRISEqhRo4betho1apCcnMyDBw8K3Gf+/PmoVCrdw8PDoyxCFUIIIUqssNpLCoUZ1f89jcyEi0QvHskbb07kk08+0b1epUoVli5dypdffom7uzsvvPBCWYZtckyq5+VJzJw5k8mTJ+ueJycnSwIjhBDCJBVVe8na0w/rkcsB+LS/H538avLwCj4jR45k5MiRBo+xPDCpnhdXV1du3Liht+3GjRs4ODhgbW1d4D5WVlY4ODjoPYQQQhiHVqtl9OjRODk5STn7AkjtpdJhUj0vgYGB7NixQ2/brl27CAwMNFJEQgghSiIsLIzQ0FDCw8OpW7cu1atXN3ZIJkVqL5UOg/a8pKSkEB0drcu8L1++THR0tO6e9ZkzZzJ48GBd+zFjxnDp0iWmTZvGuXPn+OKLL9i4cSNvvfWWIcMUQghRSmJjY3Fzc6Nt27a4urpSpYr+b+TKftuv1F4qHQZNXo4ePUqLFi1o0aIFAJMnT6ZFixbMmjULALVarVd8x8vLi19++YVdu3bRvHlzFi5cyNdffy23SQshRDkwdOhQ3nzzTeLj41EoFHh6ehIUFMT48eOZNGkS1atX132fR0REEBAQgJWVFW5ubsyYMYPs7GzdsYKCgnjzzTeZNGkSVatWpUaNGqxcuZLU1FSGDRuGvb099evX59dffzXW6T6xvNpLrir9oSFXlZLlg1pK7aViUGgfng1UASQnJ6NSqUhKSpL5L0IIUYaSkpJYunQpX331FUeOHMHc3JyXX36ZY8eOMXbsWEaMGAGAnZ0dDRo00CU7586dY9SoUYwbN445c+YAuclLVFQU06ZNo1+/fmzYsIE5c+bwr3/9iz59+hAUFMRnn33Gxo0biY+Px8bGxohn/mRyNFqpvfSQkly/JXkRQghRahYvXszixYuJi4sDcpOQ5ORkoqKidG3eeecdNm3aRExMDApF7sX6iy++YPr06SQlJWFmZkZQUBA5OTns27cPyC3OplKp6Nu3L9999x2QW17Dzc2NgwcP0qZNm7I9UVHqSnL9NqkJu0IIIcqXR3sPNAX8Hm7VqpXe85iYGAIDA3WJC0C7du1ISUnh6tWr1K5dG4BmzZrpXjc3N6datWo0bdpUty2vLlhiYmKpnpMwfZK8CCGEeCIFrc/D6b94kKVfwt7W1vaJjm9hYaH3XKFQ6G3LS340Gs0THV+UXyZV50UIIUT5kLc+z6PVYpMfZHM7JZOw0+pC923cuDEHDx7UK8B24MAB7O3tqVWrlsFiFhWHJC9CCCFKpLD1eR4Wsv0sOZqCW7zxxhv8/fffusm6P/30E7Nnz2by5MmYmcllSTyeDBsJIYQokcLW53mYOimdyMt3CnytZs2a7Nixg7fffpvmzZvj5OTEiBEjePfddw0RrqiAJHkRQghRIkWtz+PwzAuk/XWQO7u/IrG/H+Hh4QW269SpE5GRkYUep6D98u5gelgFu2FWFJMkL0IIIUrkcevuOPd5B4WZuazPIwxGBheFMHFhYWG0b98eR0dHqlWrRs+ePYmNjTV2WKISy1ufp7ByalWs7anp4iTr8wiDkeRFCBOXmprK5MmTOXr0KHv27MHMzIw+ffrI7aHCaB63Po963QxqnF2PuVnuEgEffvghw4cPx97entq1a/PVV1+VecyiYpHkRQgTk6PRcjD2Nj9FX+Ng7G169+lL3759qV+/Pn5+fnz77becOnWKs2fPGjtUUYkVtT5Pgxr2eFb7p7bLwoUL8ff35/jx47zxxhuMHTuW8+fPl3XIogKROS9CmJCHi349uHSMpD83kHXrMmaabCyqVNG7jbRp06Zs2rSJZcuWcfjwYby9vVmxYgWBgYFGPANRmQT7utHVxzXf+jzP7Zyn1+7555/njTfeAGD69Ol89tln/P777zRs2NAYYYsKQHpehDARjxb90mSl4/BMbxRKFWbV61KngQ81a9bk4MGDun3eeecdpk6dSnR0NA0aNGDAgAF6K/MKYWjmZgoC61WjZzN3AH4+eZ3kB1l6dwE9XOZfoVDg6uoqJf3FU5GeFyFMQI5Gy5xtZ/SKftk2bEfOg2RykhJw7jEJWy9vjn3wIsePH9e1mTp1Kj169AAgJCSEJk2acPHiRRo1alTGZyAqs0eXCUhQJ6M+epXu/19lt6Ay/zJnSzwNSV6EMAGf771IQnKG3rasO9e4t28NKBQkrH+HG2a5/7m+//77ujYP/6J1c3MDchepk+RFlJW8HsNHq62kZmQzdk1UvnWOhCgNMmwkhJGFnVbz2e4L+bYnbpqHJj0Fx05DqWLvjFaTexEYPHiwro0sUifKSkZGBhMmTMDFxQWlUkn79u05dDiSkO1neRB/kisLevIgLhr16klkXD1D6rl9ZN2+yr20rAJXmhbiaUjyIoQR5a0Rk2/7g2Sy71xF1bYfqtYvUnPM19QY8AEAvr6+XL58uaxDFZXctGnT2LRpE6tXryYqKor69evzr27duJpwU9fm3h/fU/XZEVjUqIdCoeDWr0vI0WiJv51mxMhFRSTDRkIYUWFrxJgp7TCzdiDlxE7M7ZzITr5J6v7vjBChELm1hpYvX05oaCjdu3cHYOXKlWzfEUbKyd+wcvMGwLHjayhrN8V9yGIexB4h8ccQak/ZzLMvBegdLzo6uqxPQVQw0vMihBEVtkaMQmFG9X9PIzPhIte/GcfdPSuZNffDMo6u/Hu0Zk5hqxyL/B7+7DaHHyMrK4t27drpXrewsMDXrxVZt//WbbN08dL9bW6XW103J+2eLBMgSp30vAhhREV9qVt7+mE9cjkAb3XxZmKXBkwb9s/F99EF6RwdHWWRuoc8egcMgJtKyexePgT7uhkxMtP36GeXmZg7TBl+PpEhdero2lW1scTG8p8JuQoz84eOkjsHy9nOUpYJEKVOel6EMKLHrREDuRfc8Z29yyymiuDRmjl5EpLSGbsmirD/v4VX5FfQZ1fF0Q3MqzD18426zy4rK4ujR4/w76CAwg4FwKTnvDE3K+pfuBAlJ8mLEEb0uDViFMDsXj7y5V8CeZOgC+qDytsWsv2sDCEVoLDPzsxSib3f89z9/VveWhjKqdNnGDVqFGlpaXzyzltM7tog37Gq21kC0KmhSxlELiobSV6EMLKi1ohZPqilDHGUUGGToFNO7ebKgp5oAXVSOpGX75R9cCausM8OoGrQUGwatuPCho/wb9WKixcvsnPnTqpWrUqAVzUAvh7yDEv6+/HDqDasGlZ0j4wQT0PmvAhhAgpbI0Z6XEqusEnQ2fduYOXh+9h2lVlRn4miiiVOXV7HqcvrLOnvxwt+NXWvBQUFFTDfqprMwRIGI8mLECYib40Y8XQKmwT94PJRnLqMeWy7yqy4n4l8dsLYZNhICGEQQUFBTJo0qczft7BJ0G6DP8PKvSEKcidBl/UdMKGhoTg6Oj62nUKhYOvWrQaPpyCPm0BurM9OiEdJ8iKEqFAeNwkajDMJul+/fly48M8yEHPmzMHPz69MY3gcU/3shHiUJC9CiArHFCdBW1tb4+Ji+nfemOJnJ8SjJHkRQhiMRqNh2rRpODk54erqypw5c3SvLVq0iKZNm2Jra4uHhwdvvPEGKSkputevXLlCr169qFq1Kra2tjRp0oQdO3YU+72Dfd3YP70zP4xqo7sDZv/0zqV68f35559xdHQkJye3UFt0dDQKhYIZM2bo2owcOZJBgwbpDRuFhoYSEhLCiRMnUCgUKBQKQkNDdfvcunWLPn36YGNjg7e3N9u2bSu1mIujLD47IZ6GJC9CCINZvXo1tra2HD58mI8//pi5c+eya9cuAMzMzFi6dClnzpxh9erV7N27l2nTpun2HTduHBkZGfzxxx+cOnWKBQsWYGdnV6L3z5sE/YJfTQLrVSv14Y4OHTpw//59jh8/DkBERATVq1cnPDxc1yYiIoKgoCC9/fr168eUKVNo0qQJarUatVpNv379dK+HhITwyiuvcPLkSZ5//nkGDhzInTtle2u3oT87IZ6GJC9PqE+fPtjZ2REXF2fsUIQwWc2aNWP27Nl4e3szePBg/P392bNnDwCTJk3i2WefxdPTk86dO/P++++zceNG3b7x8fG0a9eOpk2bUrduXXr27EnHjh2NdSoFUqlU+Pn56ZKV8PBw3nrrLY4fP05KSgrXrl3j4sWLdOrUSW8/a2tr7OzsqFKlCq6urri6umJtba17fejQoQwYMID69evz4YcfkpKSQmRkZFmemhAmTZKXJ2Rvb092djZVq1Y1dihCmIRHF0HUkpu8PMzNzY3ExEQAdu/ezXPPPUfNmjWxt7fntdde4/bt26SlpQEwYcIE3n//fdq1a8fs2bM5efJkWZ9SoR4+V+/mAfz+ezharZZ9+/bRt29fGjduzP79+4mIiMDd3R1v75It7/Dw52Zra4uDg4PucytIce9kEqKikOTlCaSlpfHTTz9Rt25d1qxZY+xwhDC6sNNq2i/Yy4CVh5i4PpoBKw9xPP4u15Iz9dopFAo0Gg1xcXH07NmTZs2asWnTJo4dO8Z///tfADIzc/cZOXIkly5d4rXXXuPUqVP4+/uzbNmyMj+3Rz16rnuTnNm5N5wvNu3GwsKCRo0aERQURHh4OBEREfl6XYrDwsJC73ne5yaEyFUmyct///tfPD09USqVtG7dusjuz9DQUN0EtryHUmlaBZF27NiBlZUV/fv3Z/369cYORwijKmwRxMxsDXtjEgtcBPHYsWNoNBoWLlxImzZtaNCgAdevX8/XzsPDgzFjxrB582amTJnCypUrDXYexVHQuVp5NCEn4wHTQz6igV9uSfy85CU8PDzffJc8lpaWuom+QoiSMXjysmHDBiZPnszs2bOJioqiefPmdOvWrcguUAcHB90kNrVazZUrVwwdZons27ePVq1aERAQQGRkJBkZGcYOSQijKGoRxDwFLYJYv359srKyWLZsGZcuXeL7779nxYoVem0mTZrEzp07uXz5MlFRUfz+++80btzYAGdRPIWdq7nSDgtnT1LPhPO3lRc5Gi0dO3YkKiqKCxcu5Ot5CQsLo3379rz//vucPn2aDh06cOTIETIyMnRz6A4ePMizzz6LjY0NzZs3z5fkhIaGUrt2bWxsbOjTpw+3b9824JkLYXoMnrwsWrSIUaNGMWzYMHx8fFixYgU2NjZ8++23he6jUCh0k9hcXV2pUaOGocN8rByNlgN/3eLTnefYc+QM5nZO1HB1IzMzk4SEBGOHJ4RRFLWQX56CFkFs3rw5ixYtYsGCBfj6+rJ27Vrmz5+v1yYnJ4dx48bRuHFjgoODadCgAV988UWpn0NxFXWuSg9f0GrIcG5E5OU7ODk54ePjg6urKw0bNtRrm5qayuTJkzly5AjPPfcchw8fJiAggLVr1+rarF27lqlTpxIdHU2DBg1ITU3VJTCHDx9mxIgRjB8/nujoaJ599lnef/99w524ECZIoTXgylmZmZnY2Njw448/0rt3b932IUOGcO/ePX766ad8+4SGhjJy5Ehq1qyJRqOhZcuWfPjhhzRp0qTA98jIyNDr+UhOTsbDw4OkpCQcHBxK5TzCTquZsfkU99KyALix4T2qVHWnVocXObd0OGfPnjXqL0IhjOWn6GtMXB/92HaPLuRXHj3pueZotEUuuHnr1i2cnZ05deoUdnZ2eHl58fXXXzNixAgAzp49S5MmTYiJiaFRo0a8+uqrJCUl8csvv+iO0b9/f8LCwrh3716pna8QZS05ORmVSlWs67dBF2a8desWOTk5+XpOatSowblz5wrcp2HDhnz77bc0a9aMpKQkPv30U9q2bcuZM2eoVatWvvbz588nJCTEIPFDbuIyZk2U3jYzGwc06Sncu3sXgJO3cpDURVRGlWkhvyc517DTakK2n9XrsXHMuoVjzBbiz53k1q1buom48fHx+PjkluZ/+G4jN7fcwnCJiYk0atSImJgY+vTpo/eegYGBhIWFPdmJCVEOmdzdRoGBgQwePBg/Pz86derE5s2bcXZ25ssvvyyw/cyZM0lKStI9/v7771KLJUejZc62M/m2W7rUI+t2PJk3r2BuX53F+2/kG9MXojKoTAv5lfRcC5vIfCb0XQ7HxDPqPx9x+PBhDh8+DPxzlxXo322kUOS+o9xtJMQ/DJq8VK9eHXNzc27cuKG3/caNG7i6uhbrGBYWFrRo0YKLFy8W+LqVlRUODg56j9ISefkOCcn5J+Na121J1q140q9Eo/RsUeCYvhCVQWVayK8k51rY5N6cB8lk37mKY9t+bLtZjQYNG3H3/3twi6tx48a6hCfPoUOHSnQMIco7gyYvlpaWtGrVSldRE3J/PezZs4fAwMBiHSMnJ4dTp07puk7LUuL9gifnWTp7YuHixYO/DmHfvFuRbYWo6CrTQn7FPdfCJveaKe0ws3bg/omdxMdd4ot1W5k8eXKJYpgwYQJhYWF8+umn/PXXX3z++ecyZCQqHYPOeQGYPHkyQ4YMwd/fn4CAABYvXkxqairDhg0DYPDgwdSsWVN3p8HcuXNp06YN9evX5969e3zyySdcuXKFkSNHGjrUfIoa47Zya0D27b+xdG/w2LZCVHTBvm509XEtcmJqRVGccy3sx4xCYUb1f0/j7u4vuf7NOBb9Xp/QlcsLrQVTkDZt2rBy5Upmz57NrFmz6NKlC++++y7z5s172lMTotwwePLSr18/bt68yaxZs0hISMDPz4+wsDDdJN74+HjMzP7pALp79y6jRo0iISGBqlWr0qpVK/7880/dRLayFODlhKuDVYFDR5Y16mFmaUvO/dt4eHhUiDF9IZ5G3kJ+lcHjzrWoHzPWnn5Yj1wOwLpRbQisV42Hb/p89AZQR0fHfNuGDx/O8OHD9bZNmTKl2PELUd4Z9FZpYyjJrVbFUdDdRo9aUcG6xoUQTydHo6X9gr0kJKUXWMBPQe5Q0/7pnStk75QQT6Ik12+Tu9vI1AT7urFiUEscbSzyvVbVxkISFyFEPpVpIrMQxiA9L8WUo9FyKPY2By/dAnK7jNvUrSZfPkKIQhVU58VNpWR2Lx/50SPEI0py/ZbkRQghDOhxFXaFELlMpsKuEEJUdpVpIrMQZUXmvAghRCUXHh6OQqEw2NpIBw4coGnTplhYWOitcyfEk5LkRQghKpmgoCAmTZpUZu83efJk/Pz8uHz5MqGhoWX2vqLikuRFCFGhhYWF0b59exwdHalWrRo9e/YkNjYWyF1PaPz48bi5uaFUKqlTp46uYKYoPbGxsXTu3JlatWrh6Oho7HBEBSDJixCiQktNTWXy5MkcPXqUPXv2YGZmRp8+fdBoNCxdupRt27axceNGzp8/z9q1a/H09DR2yAY1dOhQIiIiWLJkCQqFAoVCQVxcHADHjh3D398fGxsb2rZty/nz5/X2/emnn2jZsiVKpZK6desSEhJCdnY2GRkZTJgwARcXF5RKJe3bt+fIkSPExcWhUCi4ffs2w4cPR6FQSM+LKB3aCiYpKUkLaJOSkowdihCiFKWnp2vffPNNrbOzs9bKykrbrl07bWRkpFar1Wp///13LaDdvXu3tlWrVlorpbW2YbNW2h9+O6jNztHojrF161Zt06ZNtYC2Zs2a2oCAAG1QUJBWo9EU9rYVzr1797SBgYHaUaNGadVqtVatVmt3796tBbStW7fWhoeHa8+cOaPt0KGDtm3btrr9/vjjD62Dg4M2NDRUGxsbq/3tt9+0np6e2jlz5mgnTJigdXd31+7YsUN75swZ7ZAhQ7RVq1bV3rp1S6tWq7UODg7axYsXa9VqtTYtLc2IZy9MWUmu35K8CCHKhcIukLdv39YlLw2btdT6jFyodRvxhdaqVhOtVc3G2uaTQ7Wdur+gdXNz0wJaS0tLLaB9//33te7u7lpra2utt7e39s0339Tu3LnT2KdZJjp16qSdOHGi7vnDyV+eX375RQtoHzx4oNVqtdrnnntO++GHH+od5/vvv9fWqFFDa2FhoV27dq1ue2Zmptbd3V378ccfa7VarValUmlXrVpluBMSFUJJrt8ybCSEMHmpqaksX76cTz75hO7du+Pj48PKlSuxtrbmm2++0bW717gvqdUaYlm9Nqo2L5FxLYbTq97hcEw8No7VmDRpElFRuct9NGnShAULFqBSqZg3bx4PHjzglVde4aWXXjLWaRpUjkbLwdjb/BR9jeQHWfnWSwJo1qyZ7m83t9wieomJiQCcOHGCuXPnYmdnh52dHTa2dowYOZIbN26QlZVFu3btdPtaWFgQEBBATEyMgc9KVFZS50UIYZIeLu6WfC22yAtky1b+udtcvHSvm9vlLpaac/cazt3f5MrW+axYsYIvv/wSgAEDBgCQnp5Or1696NevHy+99BLBwcHcuXMHJ6eKs9jqo5V+E9TJqI9epftptV6lXwuLf5ZBUShyC+lpNBoAUlJSCAkJobpve5btvUji/dwFa7NuX+XmprmEn09kSJ06ZXVKopKT5EUIYXIevdhmJl4GKPQCeU6dDIDCzPyhrbkXX4WVHfdP7CQ7I43nX+rP5bPRnDp1ikWLFvH333/j7OxMXFwcVapU4X//+x+urq4V6o6YsNNqxq6J0lsgUmFuQWp6JmPXRLF8UEsKXwP7Hy1btmTPoeNcuNMEbZVqWFTN3W5u6wTmVZj6+UZquNci2NeNrKwsjhw5Uqa3Y4vKRYaNhBAmJe9i+/B6QFUc3XQXyLDTagDdBdLHx4e7DzILPZ5T19fJTLgIOdns/XUby5YtA3KHRby8vPjuu+8ICAjgmWeeIS4ujh07dmBmVjG+GnM0WkK2n823snUVlQsZ6vNkJd3g3fUHycrOeeyx3nn3PXb99D/u7l9H5s0rZN36m9SzESQd+h/2fs9z9/dveWthKKdOn2HUqFGkpaUxYsQIw5yYqPSk50UIYTIKu9iaWSofukA6UXPKv1n46Se6C+S6HX8UekxlLR/sRi7nwaVj3N4yj/DwcE6fPo2ZmRnp6en06NGDkLnzdENU6fZKcjTaCrH+UOTlO3pJYB6HgL7c+mUR179+g2vZGUQ4LHvssao2eAbnF2eR9Od6kg9vAjNzLKrVwq7Zv7Dz7QxoubDhI/zXhfDMM/7s3LmTqlWrGuCshJDkRQhhQgq72AJUDRpKYRfIRm6PWYEWqNuiHauGbeeD9+exYMECLCwsaNSoEW26v0z7BXsr5MrPifcL/iwtnGri9tpCAFJO7eazee/km8Dr5+en2zZ06FBiriRg3fpNrOu2KvCYTl1ex6nL6/ynYzVe7xGoN3/GUMsOiMqrYvSNCiEqhMIutgCKKpY4dXkdjwnr2Hg4lv379/PMM88A8FznZ/n11HXMlXbk9ZdY1qhLnek/Y6GqAcDsXj483z2YAwcOkJaWRlJSEiHfbGVbeuN8CVNCUjpj10TphqjKKxf74sxmgcd1Mi1ZsoR5i74o1rGq21oVq50QT0OSFyGEySjuxbagdsG+biwf1BJXlf5rriolywe1zNeLUtgQFUDSse0krP8PIdvPkqMpqEX5EODlhJtKSWG5iQJQWVs8dohMpVLxXHOvxx7LTaWkuYfjkwcsRDFJ8iKEMBnFudi6qZQEeBV8G3Owrxv7p3fmh1FtWNLfjx9GtWH/9M4FDv8UNUSleZBM1t0E1EnpRF6+84RnYzw///xz7h1TWg2ze/mQceMSVxb05G54qK7N7V+XcnP7p/T2cwdg586dNG7cGDs7O4KDg1Gr/+l1Gjp0KC/27cPsXj65G7Qakg7/yLUvR3Hl095c/WIY9/7cwOxePrpE6NKlSzz77LPY2NjQvHlzDh48WFanLyoBSV6EECbD3Eyhu0A+msDkPX/4AlnYMQLrVeMFv5oE1qtWaNuihqgc2w+k1thvH9vOVHXo0IH79+9z/Phxgn3d+HeNe1SxUZEef0rXJvvaGUa+0oOmtRxJS0vj008/5fvvv+ePP/4gPj6eqVOn5jtuXu9W5qG1JB/6EVXb/riPWE7DAf9hyHPN9JLEd955h6lTpxIdHU2DBg0YMGAA2dnZZXL+ouKTCbtCCJOSd4F8uM4L5A7/lOYk2qcZojJFDxf1c7FX4ufnR3h4OP7+/tw4H8Xs/7zN3JC5fPRvb2y0mbyw4BrjXn2BAwcOkJWVxYoVK6hXrx4A48ePZ+7cuQW+T7s6dtw5vJVpsz6izfMv42Kf2xP2aJI4depUevToAUBISAhNmjTh4sWLNGrUyLAfhKgUJHkRQpicYF83uvq46l2MC7pAPo28IaqEpPQC570oyE2YChuiMiWPFvUDyLD24n/bdzJlyhT27dvH/Pnz+fF//8Mx+RJ37tzB3d0db29vDhw4gI2NjS5xgdwaOHnLAjwqJiaGjIwMRg94AS+vmoXGVNhSA5K8iNIgyYsQwiTlDf8Y8vize/kwdk0UCtCvQPv///u4ISpTUFAFXQBNDR+O/LKILzbt1t0WHhQURHh4OHfv3qVTp066tg/f1gy5SwMUtPYRgLW1dbHiKmqpASGelsx5EUJUWiW9Q+lpaLVaRo8ejZOTEwqFgujo6Kc+ZlF3TFl6NEGb+YA5H35Cx465iUpe8hIeHk5QUNATvae3tzfW1tbs2bPnyQMX4ilJz4sQolIriyEqgLCwMEJDQwkPD6du3bpUr179qY9Z1B1T5ko7LJw9uRW9hzp9FgDQsWNHXnnlFbKysvR6XkpCqVQyffp0pk2bhqWlJe3atePmzZucOXNGlgMQZUaSFyFEpWfoISqA2NhY3NzcaNu27RPtr9VqycnJoUqVf762H3cnlNLDl6zES9TxzS3m5+TkhI+PDzdu3KBhw4ZPFAfAe++9R5UqVZg1axbXr1/Hzc2NMWPGPPHxhCgphbawgc1yKjk5GZVKRVJSEg4ORZcMF8IYgoKC8PPzY/HixcYORZSRoUOHsnr1at3zOnXqcP78ed5++23Wr19PcnIy/v7+fPbZZ7qqweHh4Tz77LPs2LGDd999l1OnTvHbb7/pDfccjL3NgJWHHvv+P4xqY/DkTIinVZLrt/S8CFHGNm/enG+CpKjYlixZQr169fjqq684cuQI5ubmTJs2jU2bNrF69Wrq1KnDxx9/TLdu3bh48SJOTv/c4TRjxgw+/fRT6tatm2+hw4p0x5QQJSETdoUoY05OTtjb2xs7DFGGVCoV9vb2mJub4+rqio2NDcuXL+eTTz6he/fu+Pj4sHLlSqytrfnmm2/09p07dy5du3alXr16ekkNlE5RPyHKI0lehChjQUFBTJo0CYAvvvgCb29vlEolNWrU4KWXXjJucKLU5Gi0HIy9zU/R1zgYexvNQyP0sbGxZGVl0a5dO902CwsLAgICiImJ0TuOv79/ke9TlndMCWEqZNhICCM5evQoEyZM4Pvvv6dt27bcuXOHffv2GTssUQoKKhrH6b94kJVT4mPZ2to+tk1Z3TElhKmQ5EUII4mPj8fW1paePXtib29PnTp1aNGihbHDEk+psKJxyQ+ySU7JJOy0mg716mFpacmBAweoU6cOAFlZWRw5ckTXK1dSZXHHlBCmQoaNhCgDDw8hJD/IQqvV0rVrV+rUqUPdunV57bXXWLt2LWlpacYOVTyFoorG5QnZfhaltQ1jx47l7bffJiwsjLNnzzJq1CjS0tKkVooQxSA9L0IY2KNDCAnqZNRHr9L9SgpRUVGEh4fz22+/MWvWLObMmcORI0dwdHQ0btDiiRRVNC6POimdyMt3+Oijj9BoNLz22mvcv38ff39/du7cme+OIiFEfmXS8/Lf//4XT09PlEolrVu3JjIyssj2//vf/2jUqBFKpZKmTZuyY8eOsghTiFKXN4Tw6AUtNSObsWui2H3uJl26dOHjjz/m5MmTxMXFsXfvXiNFK55WUUXjHJ55gVpjv9W1UyqVLF26lJs3b5Kens7+/ft1NV4gd2K3VquVRFaIAhg8edmwYQOTJ09m9uzZREVF0bx5c7p161boiqV//vknAwYMYMSIERw/fpzevXvTu3dvTp8+behQhShVjxtCSLsYyZiZH3As6jhXrlzhu+++Q6PRPFXlU2FcLvbKxzcqQTshRMEMnrwsWrSIUaNGMWzYMHx8fFixYgU2NjZ8++23BbZfsmQJwcHBvP322zRu3Jh58+bRsmVLPv/8c0OHKkSpetwQgkJpS8KJCJ7t3JnGjRuzYsUKfvjhB5o0aVKGUYrSlFc0rrB7fBSAmxSNE+KpGTR5yczM5NixY3Tp0uWfNzQzo0uXLhw8eLDAfQ4ePKjXHqBbt26Fts/IyCA5OVnvIYQpKGwIwfXVj3DqMhplrSa4vvoR34efJi0tjRMnTvDKK6+UcZSiNEnROCHKhkGTl1u3bpGTk0ONGjX0tteoUYOEhIQC90lISChR+/nz56NSqXQPDw+P0gleiKckQwiVkxSNE8Lwyv3dRjNnzmTy5Mm658nJyZLACJMg685UXlI0TgjDMmjyUr16dczNzblx44be9hs3buDq6lrgPq6uriVqb2VlhZWVVekELEQpyhtCGLsmCgXoJTAyhFDxSdE4IQzHoMNGlpaWtGrVij179ui2aTQa9uzZQ2BgYIH7BAYG6rUH2LVrV6HthTBlMoQghBClz+DDRpMnT2bIkCH4+/sTEBDA4sWLSU1NZdiwYQAMHjyYmjVrMn/+fAAmTpxIp06dWLhwIT169GD9+vUcPXqUr776ytChimIKDQ1l2LBhaLVF1REVeWQIQQghSpfBk5d+/fpx8+ZNZs2aRUJCAn5+foSFhekm5cbHx2Nm9k8HUNu2bVm3bh3vvvsu//nPf/D29mbr1q34+voaOlRRiByNVu/Ca2/vILVISkiGEIQQovQotBXs53NycjIqlYqkpCQcHByMHU65V9DquG4qJbN7+ciQhxBCiFJTkuu3LMwoClVYafuEpHTGroki7LTaSJEJISq7oKCgJ16BW5R/kryIAhVV2j5vW8j2s+RoKlTHnRCinNi8eTPz5s0DwNPTk8WLFxs3IFGmJHkRBSqqtH3ahT+5unKMbnVcIYQoa05OTtjb2xs7DGEkkryIAhW1Oq4mI43sO1cf204IIQwlb9goKCiIK1eu8NZbb6FQKFAocu/iu3LlCr169aJq1arY2trSpEkTduzYYeSoRWkp9xV2hWEUVbLermkX7Jp2eWw7IYQwtM2bN9O8eXNGjx7NqFGjdNvHjRtHZmYmf/zxB7a2tpw9exY7OzsjRipKkyQvokBS2l4IYWoeLtuQ/CALrVaLk5MT5ubm2Nvb61Vij4+P58UXX6Rp06YA1K1b11hhCwOQ5EUUSErbCyFMyaNlGxLUyaiPXqV7IXc9TpgwgbFjx/Lbb7/RpUsXXnzxRZo1a1aWIQsDkjkvolBS2l7kCQ8PR6FQcO/evULbzJkzBz8/vzKLSVQehZVtSM3IZuyaKB5k5eTbZ+TIkVy6dInXXnuNU6dO4e/vz7Jly8oqZGFgkryIIgX7urF/emd+GNWGJf39+GFUG/ZP7yyJSwX3JDU0pk6dmm9dMiGeVlFlG/IkZ2jJys7Ot93Dw4MxY8awefNmpkyZwsqVKw0XaCWzdu1a7OzsdI99+/aV6fvLsJF4LCltL4oj70tMiNJUVNkG+P8hbXsXtoXt4dUBA7CysqJ69epMmjSJ7t2706BBA+7evcvvv/9O48aNyyzuiipv3pG5pz+rfvqdZh6OmJspqFmzZpnGIT0vQgg9Q4cOJSIigiVLluhuPY2LiwPg2LFj+Pv7Y2NjQ9u2bTl//rxuv0eHjcLDwwkICMDW1hZHR0fatWvHlStXyvhsRHlXnHIMju0HcjX+CvXq1cPZ2RmAnJwcxo0bR+PGjQkODqZBgwZ88cUXhg63Qgs7rab9gr0MWHmIGdsv8vauRIZuiudiui3W1tZlGov0vAgh9CxZsoQLFy7g6+vL3LlzAThz5gwA77zzDgsXLsTZ2ZkxY8YwfPhwDhw4kO8Y2dnZ9O7dm1GjRvHDDz+QmZlJZGSkrgaHEMVVWDkG11c/0v1tVbMR637dp9dDLPNbSlfevKNHh+/ylosp63mQkrwIIfSoVCosLS2xsbHR3Xp67tw5AD744AM6deoEwIwZM+jRowfp6ekolfoXmOTkZJKSkujZsyf16tUDkC578USkbIPxPW65GAW5y8V09XEtsztQZdhICAHkfkEdjL3NT9HXdDU0HvXwraZubrm/shITE/O1c3JyYujQoXTr1o1evXqxZMkS1GpZyFOUXF7ZBvinTEMeKdtQNooz76isl4uR5EUIoTeWPXF9NGfVyWw8ejXfyuEWFha6v/OGgDQaTYHHXLVqFQcPHqRt27Zs2LCBBg0acOjQIcOdhKiwpGyDcRV3GZiyXC5Gho2EqOQKGstWmFuQmp6pG8t+0kUgWrRoQYsWLZg5cyaBgYGsW7eONm3alEbYopIJ9nWjq4+rrsKui33uUJH0uBhecZeBKcvlYqTnRYhKrLCx7CoqFzLU58lKusG76w+SlZ2/CFhRLl++zMyZMzl48CBXrlzht99+46+//pJ5L+Kp5JVteMGvJoH1qkniUkby5h0V9mkrALcynnckyYsQlVhhY9kOAX1BYcb1r9/g2AcvEhEVU6Lj2tjYcO7cOV588UUaNGjA6NGjGTduHK+//npphS6EKCOmOO9IoS1oVl45lpycjEqlIikpCQcHB2OHI4RJ+yn6GhPXRz+23ZL+frzgV7ZFqIQQpuXR9aUgt8dldi+fUpl3VJLrt8x5EaISM5Wx7KCgIPz8/Fi8eLFB30cI8eRMad6RDBsJUYmZ4lj2k4iLi0OhUBAdHW3sUISo0Exl3pEkL0JUYqU5lv0kizkKIcSTkORFiEqutGpobN68mXnz5gHg6elZ4iGg7Oxsxo8fj0qlonr16rz33nu6QnkKhYKtW7fqtXd0dCQ0NBQALy8vIPfWbIVCQVBQUIneWwhRvsicFyFEqYxlOzk93dDS6tWrGTFiBJGRkRw9epTRo0dTu3ZtRo0a9dh9IyMjCQgIYPfu3TRp0gRLS8unisWQZH6PEE9PkhchBPDPWPaTyrsoR0dHc+XKFd566y3eeustgAKXGniUh4cHn332GQqFgoYNG3Lq1Ck+++yzYiUveSsJV6tWTbcekxCi4pJhIyFEqdq8eTO1atVi7ty5qNXqQtc0enQtpdatW+utOh0YGMhff/1FTk7JCuSZsqFDhxIREcGSJUtQKBQoFAri4uKIiIggICAAKysr3NzcmDFjBtnZ2cYOVwiTJT0vQognlqPR6oaa8hZzdHJywtzcHHt7+0J7QR6tF5GgTuZqjpqw0+oC59goFIp8vTdZWVmlf0IGtmTJEi5cuICvry9z584FICcnh+eff56hQ4fy3Xffce7cOUaNGoVSqWTOnDnGDVgIEyXJixDiiRSUgKiPXqX76aJXjy5oLSWAe3ExurWUgn3dOHToEN7e3pibm+Ps7KzXg/PXX3+Rlpame543x8XUe2lUKhWWlpbY2NjoErt33nkHDw8PPv/8cxQKBY0aNeL69etMnz6dWbNmYWYmHeRCPEqSFyFEiRWWgKRmZDN2TRQPsgpOIgpbSwkg+/5N7uxZyYzM3txuZcmyZctYuHAhAJ07d+bzzz8nMDCQnJwcpk+frrfCtYuLC9bW1oSFhVGrVi2USiUqlaqUzvbpFdRDlScmJobAwEC9IbN27dqRkpLC1atXqV27tjFCFsKkSUovhCiRohKQPMkZWrIKmLNR2FpKALZNOqPJzuTkf8cxdtw4Jk6cyOjRowFYuHAhHh4edOjQgVdffZWpU6diY2Oj27dKlSosXbqUL7/8End3d1544YWnOsfSFHZaTfsFexmw8hAT10dzVp3MxqNXCXtMD5UQonDS8yKEKJGiEhAgN6mxd2Fb2B5eHTAAKysrqlevDkDi/YL3c331I93f1bqNy7eWkru7Ozt37tTb5969e3rPR44cyciRI0t2MgZWUA+VwtyC1PRM3RBZ48aN2bRpE1qtVtf7cuDAAezt7alVq5ZxAhfCxEnPixCiRApLQB7m2H4gV+OvUK9ePd1tzGA6aymVhcJ6qKqoXMhQnycr6Qbvrj/I62PG8vfff/Pmm29y7tw5fvrpJ2bPns3kyZNlvosQhZD/MoQQJVJYYuH66kc4dckd5rGq2Yh1v+4jPT1db35H2pWTXFnQE016SoHHKC9rKRVHYT1UDgF9QWHG9a/f4NgHL3L00k127NhBZGQkzZs3Z8yYMYwYMYJ3333XCFELUT4YNHm5c+cOAwcOxMHBAUdHR0aMGEFKSsFfWnmCgoJ09Q/yHmPGjDFkmEKIEijJYo6Prnf0cMXep1lL6UmWHyhrhfVQWTjVxO21hdSesok603/GXOVCp06diIyMJCMjA7VazUcffUSVKjKqL0RhDJq8DBw4kDNnzrBr1y5+/vln/vjjD90EvKKMGjVKV9xKrVbz8ccfGzJMIUQJlMZijoteaf7UaymZuso0RCZEWTNY8hITE0NYWBhff/01rVu3pn379ixbtoz169dz/fr1IvfNq4GQ93BwcDBUmEKIJ1CcxRwLqyYLUPXBNbI2TSdhycsof53NR89VY//0zgT7uhEbG8sLL7xAjRo1sLOz45lnnmH37t269wgKCtItP5B3XFNUkh4qIUTJGCx5OXjwII6Ojvj7++u2denSBTMzMw4fPlzkvmvXrqV69er4+voyc+ZMvWJUj8rIyCA5OVnvIYQwvGBfN/ZP78wPo9qwpL8fP4xqo0tAILeabGBgoF5PqoeHB5BbmG3hwoVEHTuKi8qGZXOm6HpqUlJSeP7559mzZw/Hjx8nODiYXr16ER8fDxR/+QFjK40eKiFEwQw2qJqQkICLi4v+m1WpgpOTEwkJCYXu9+qrr1KnTh3c3d05efIk06dP5/z582zevLnA9vPnzyckJKRUYxdCFE9Bizk+XJAtPUeBtbW1rprsuXPnAPjggw/o1KkTADNmzKBHjx6kp6ejVCpp3rw5zZs31x1v3rx5bNmyhW3btjF+/PhiLT9gKvJ6qB6uRAy5PVSze/lUmCEyIcpaiZOXGTNmsGDBgiLbxMTEPHFAD8+Jadq0KW5ubjz33HPExsZSr169fO1nzpzJ5MmTdc+Tk5N1v+6EEGWrqCUDHr5QN2vWTPe3m1vu9sTERGrXrk1KSgpz5szhl19+Qa1Wk52dzYMHD3Q9L+VNsK8bXX1cdQmdi33uUJH0uAjx5EqcvEyZMoWhQ4cW2aZu3bq4urqSmJiotz07O5s7d+6U6NdS69atAbh48WKByYuVlRVWVlbFPp4QwjAet2TA8kEtyZsh83Bp/7w5KxqNBoCpU6eya9cuPv30U+rXr4+1tTUvvfQSmZmZZXAWhlFQD5UQ4smVOHlxdnbWKzpVmMDAQO7du8exY8do1aoVAHv37kWj0egSkuKIjo4G/vl1JoQwPYUVZFOYW4A2NykJ2X6W91s/fprdgQMHGDp0KH369AFy58DkTfTNY2lpafKLMAohDMdgE3YbN25McHAwo0aNIjIykgMHDjB+/Hj69++Pu7s7ANeuXaNRo0ZERkYCEBsby7x58zh27BhxcXFs27aNwYMH07FjR71uZiGEaSmsINvD1WSvqm9w9vq9xx7L29ubzZs3Ex0dzYkTJ3j11Vd1vTJ5PD09+eOPP7h27Rq3bt0qrdMQQpQTBq3zsnbtWho1asRzzz3H888/T/v27fnqq690r2dlZXH+/Hnd3USWlpbs3r2bf/3rXzRq1IgpU6bw4osvsn37dkOGKYR4SoUVZHu4muzVZQP561LcY4+1aNEiqlatStu2benVqxfdunWjZcuWem3mzp1LXFxcvuUHhBCVg0L7cO3uCiA5ORmVSkVSUpLUhxGijByMvc2AlYce2+6HUW1k7ocQokAluX7L2kZCiKcmBdmEEGVJkhchxFOTgmxCiLIkyYsQolQUZ8kAIYQoDbJsqRCi1EhBNiFEWZDkRQhRqqQgmxDC0GTYSAghhBDliiQvQgghhChXJHkRQgghRLkiyYsQQgghyhVJXoQQQghRrkjyIoQQQohyRZIXIYQQQpQrkrwIIYQQolyR5EUIIUSp0mq1ZGdnGzsMUYFJ8iIAGDp0KHPmzAFAoVAQFxdn1HiEEKYlIyODCRMm4OLiglKppH379hw5cgSA8PBwFAoFv/76K61atcLKyor9+/cbOWJRkUnyIoQQ4rGmTZvGpk2bWL16NVFRUdSvX59u3bpx584dXZsZM2bw0UcfERMTQ7NmzYwYrajoZG0jIYQQRUpNTWX58uWEhobSvXt3AFauXMmuXbv45ptveOaZZwCYO3cuXbt2NWaoopKQ5EUIIUQ+ORqtbnXw5GuxZGVl0a5dO93rFhYWBAQEEBMTo0te/P39jRWuqGQkeREAhIaG6v7WarXGC0QIYXRhp9WEbD+LOikdgMzEywCEn09kSJ06he5na2tbJvEJIXNeKrEcjZaDsbf5KfoaB2Nvk6ORpEWIyi7stJqxa6J0iQtAFUc3MK/C1M83EnZaDUBWVhZHjhzBx8fHWKGKSkx6XiqpR39ZAbiplMzu5UOwr5sRIxNCGEuORkvI9rM8+jPGzFKJvd/z3P39W95a6ETNKf9m4aefkJaWxogRIzhx4oRR4hWVl/S8VEIF/bICSEhKZ+yaKN0vKyFE5RJ5+U6+74U8VYOGYtOwHRc2fIR/q1ZcvHiRnTt3UrVq1TKOUgjpeal0CvtlBaAFFEDI9rN09XHF3ExRxtEJIYwp8X7BiQuAooolTl1ex6nL6yzp78cLfjV1rwUFBclcOVGmpOelkinqlxXkJjDqpHQiL98ptI0QomJysVeWajshDEWSl0qmqF9WT9JOCFFxBHg54aZSUlifq4LcuXEBXk5lGZYQ+UjyUsnILyshRGHMzRTM7pV799CjCUze89m9fGRIWRidJC+VjPyyEkIUJdjXjeWDWuKq0v8B46pSsnxQS7kbUZgEmbBbyeT9shq7JgoF6E3clV9WQgjITWC6+rjqKuy62Of+oJHvBWEqFNoKNkU8OTkZlUpFUlISDg4Oxg7HZEmdFyGEEKakJNdv6XmppOSXlRBCiPJKkpdKzNxMQWC9asYOQwghhCgRmbArhDA5QUFBTJo0ydhhCCFMlCQvQgghhChXDJa8fPDBB7Rt2xYbGxscHR2LtY9Wq2XWrFm4ublhbW1Nly5d+OuvvwwVohBCCCHKIYMlL5mZmbz88suMHTu22Pt8/PHHLF26lBUrVnD48GFsbW3p1q0b6elS7VWIyiY7O5vx48ejUqmoXr067733Hlqtljlz5tC8eXOmTp1KzZo1sbW1pXXr1oSHhxs7ZCFEGTHYhN2QkBAAQkNDi9Veq9WyePFi3n33XV544QUAvvvuO2rUqMHWrVvp37+/oUIVQpig1atXM2LECCIjIzl69CijR4+mdu3aAFy9epWDBw+yfv163N3d2bJlC8HBwZw6dQpvb28jRy6EMDSTudvo8uXLJCQk0KVLF902lUpF69atOXjwYKHJS0ZGBhkZGbrnycnJBo9VCPHkgoKCaNasGUqlkq+//hpLS0vGjBnDe7NmE3n5DmcuxBJ94iSpqal8++23XL9+nWXLlvHmm28ye/Zs1Go1AH/++ScdO3Zk1apVTJ06lbCwMFatWsWHH35o5DMUQhiayUzYTUhIAKBGjRp622vUqKF7rSDz589HpVLpHh4eHgaNUwjx9FavXo2trS2HDx/m448/Zu7cufiO/IT+X/3JG0MHkJyWgapha+au+IFLly7Rr18/AgMDSUxMpE+fPgBYW1tjbW3NuHHjsLOzIyIigtjYWCOfmRCiLJSo52XGjBksWLCgyDYxMTE0atToqYIqiZkzZzJ58mTd8+TkZElghDBxzZo1Y/bs2QDEZthh4Vqf62ePoKyTSdbNOCxd65NtYceSkzDjvYWM6R1Ez549USgUmJnl/uY6fvw45ubmese1s7Mr83MRQpS9EiUvU6ZMYejQoUW2qVu37hMF4urqCsCNGzdwc/unPP2NGzfw8/MrdD8rKyusrKye6D2FEGUjR6PVVXNOfpBFm1bNddtDtp/F3NaJnLR7ZN3+G3MHZxRVLMm4fgGA0LPZODo6Eh4ejre3t+77ITExkQ4dOhjtnIQQxlOi5MXZ2RlnZ2eDBOLl5YWrqyt79uzRJSvJyckcPny4RHcsCSFMy6PraCWok1GfuMG/T6tRWVvmblco4JFl1rLv3+T2npVk+nUnPSOTXbt2sXTpUtRqNVWrVmXw4MEsXLiQFi1acPPmTfbs2UOzZs3o0aOHMU5TCFGGDDbnJT4+nujoaOLj48nJySE6Opro6GhSUlJ0bRo1asSWLVsAUCgUTJo0iffff59t27Zx6tQpBg8ejLu7O7179zZUmEIIAwo7rWbsmii9BUABUjOyGbsmil1n9eezWVTzICf5JtrsTGybdEabncn11ZNIf5DGwIEDGT16NJaWlri7uzN48GCmTJlCw4YN6d27N0eOHNHdjSSEqNgMdrfRrFmzWL16te55ixYtAPj9998JCgoC4Pz58yQlJenaTJs2jdTUVEaPHs29e/do3749YWFhKJVKQ4UphDCQvCGhopat/yn6ut5zpacfFs6eKMwtsGvWFTQ5ZCb8RZPaLnz77bcAeHp6EhcXR58+fXjzzText7eXoWMhKhmFVqst6rul3CnJktpCCMM5GHubASsP5duesG4Gli51ceoyGgAnWwvOfT8bMytbqvd4i+zkRO7s+pL0KydAoaBqg2c49dsG3N1y58VlZGQwcOBA9uzZw71791i1atVj5+IJIUxfSa7fkrwIIQzip+hrTFwf/dh2I9p58u2BOAC9XhrF///v8kEtCfZ1e3Q3IUQFU5Lrt8nUeRFCVCwu9sUb7u3i48ryQS1xVem3d1UpJXERQhTIZCrsCiEqlgAvJ9xUShKS0guc96IgN0EJ8HLC3ExBVx9X3e3ULvb/bBdCiEdJ8iKEMAhzMwWze/kwdk0UCgoeEprdy0eXoJibKQisV62swxRClEMybCSEMJhgXzcZEhJClDrpeRFCGFSwr5sMCVVg4eHhPPvss9y9exdHR8cnPo5CoWDLli1S10sUiyQvQgiDkyGhiiMoKAg/Pz8WL15cqsfNq5wsRHHIsJEwqM6dO/Pyyy8DuV96oaGhAGzdupWNGzcaMTIhhClxdXUtsthgVlZWGUYjTJ0kL8KgnJ2d2bt3L7///rve9sDAQN55551824UQpmvo0KFERESwZMkSFAoFCoWCuLg4AI4dO4a/vz82Nja0bduW8+fP6+37008/0bJlS5RKJXXr1iUkJITs7Gzd6wqFgq1btwIQFxeHQqFgw4YNdOrUCaVSydq1a8vqNEU5IMmLMChra2sGDBjA+PHjycjI0G2vUaMGP//8M+PHj0etVhsxQiFEcS1ZsoTAwEBGjRqFWq1GrVbj4eEBwDvvvMPChQs5evQoVapUYfjw4br99u3bx+DBg5k4cSJnz57lyy+/JDQ0lA8++KDI95sxYwYTJ04kJiaGbt26GfTcRPkiyYswuOrVq3PmzJl8XcINGzbkzJkzuLnJHSei4gsPD0ehUHDv3j1jh1IiORotB2Nv81P0Nc7eysbC0hIbGxtcXV1xdXXF3NwcgA8++IBOnTrh4+PDjBkz+PPPP0lPz12QMyQkhBkzZjBkyBDq1q1L165dmTdvHl9++WWR7z1p0iT69u2Ll5eXfE8IPTJhV5SqHI1W766Sb75dpburJDw83LjBCVGGDDWxtSyFnVYTsv2s3qrgd+LvUtUjNV/bZs2a6f7OSzQSExOpXbs2J06c4MCBA3o9LTk5OaSnp5OWloaNjU2B7+/v719apyIqGEleRKkp6IvOTaVkdi8fqechxBMyVhIUdlrN2DVR+aojZ2Zr2BuTSNhptd5/1xYWFrq/FYrcHywajQaAlJQUQkJC6Nu3b773USoLX0bC1tb2Kc5AVGQybCRKRd4X3cOJC0BCUjpj10QRdlrmtYjKozQntsbFxemSgOHDh9OzZ0+99llZWbi4uPDNN9+UWvw5Gi0h288WvKyDuQVoNYRsP0uOpnjr+rZs2ZLz589Tv379fA8zM7kMiZKTfzXiqRX1Raf9/0dJvuiEKO9Kc2LrjRs3iIyMBGDkyJGEhYXpTXL/+eefSUtLo1+/fqUWf+TlO/l+iOSponIhQ32ev+Ov8Nuxv3SJVVFmzZrFd999R0hICGfOnCEmJob169fz7rvvllrMonKR5EU8taK+6PKok9KJvHynjCISwjjyJreGX04hPUeBtbX1U09s9fT05PTp00ybNk3X6zJ48GDde7777rtYWFhQo0YNPDw8eOONN0hJSQEgOTkZa2trfv31V704t2zZgr29PWlpaQD8/fffvPLKKzg6OuLk5MSkEa+SnXSjwHN0COgLCjOuf/0Gzwc0JD4+/rGfS7du3fj555/57bffeOaZZ2jTpg2fffYZderUKeEnLEQuSV7EU0u8X3TikmfX2QQDRyKE8YSdVtN+wV4GrDzExPXRnFUns/Ho1XxDpoVNbAU4ceIEc+fOxc7OTve4cOECqampWFhYcPjwYV555RV2797Nrl27uHHjBufOnWPOnDmcOXOG1atXs3fvXqZNmwaAg4MDPXv2ZN26dXoxrF27lt69e2NjY0NWVhbdunXD3t6effv2ceDAARxV9tzYOBttTv7CcBZONXF7bSG1p2ziz4u3GDp0KFqtVm9pAD8/P7RaLZ6enrpt3bp148CBA6SlpZGUlMThw4cZNWqU7nWtVqtbGsDT0xOtVoufn1+J/38QlYNM2BVPzcW+8Al3D/sp+jrv9PCRNW1EhVPY5NbUjGzGroli+aCW5P1X8riJrbPnzKGef2dup2ZQzdaKT/4zHk1ODvPmzcPMzIxly5axfv16vvvuO/z8/KhXrx4TJ04Eci/677//PmPGjOGLL74AYODAgbz22mu6u3qSk5P55Zdf2LJlCwAbNmxAo9Hw9ddf6+LZumEtdg4qMuJPofRqme98FeQurhng5VRKn6AQJSPJi3hqAV5OONlacCe16PLdt1Mzibx8R9a4ERVKYXO+8ia2Qu6cr/dbP76ju24jXxb9Lxybu766bXduptMpoIVuYmu1atVwc3MjMjKS6OhoOnTowHPPPce5c+dITk4mOztb7xbk559/HgsLC7Zt20b//v3ZtGkTDg4OdOnSBcjt7bl48SL29vZ6sWizM8m6l4A16J1b3k+P2b3kh4gwHhk2Ek/N3ExBH7+axWpb3CEmIcqLwuZ85U1szUq6wVX1Dc5ev1fkccJOq7nt3Yubx3dxb/86Mm9eIevW3zxIvsNvEX/qDT/VqVOHixcvEhMTw5o1a2jWrBmbNm3i2LFj/Pe//wUgMzMTAEtLS1566SXd0NG6devo168fVark/nZNSUmhVatWREdH6z0uXLjAV3Mm4KrS71l1VSlZPqillD8QRiU9L6JUdPFx5ZsDcY9tV9whJiHKi8IScoeAvtz6ZRHXv34DbXYGf4UsKvQYeb03yrqtcHlxFkl/rif58CYwMwe0mLs2IGT7Wbr6uGJupsDZ2Rlra2saNmzIqVOnWLhwoa5npqAFTwcOHEjXrl05c+YMe/fu5f3339e91rJlSzZs2ICLiwsODg56+9WvD71bN9ArPBng5SQ9LsLopOdFlIoALyfcVEoK+0pTkFuwTsbIRUVTWEL+8MTWOtN/ZsDAwYVObE3UOuh6b6zrtsJ10CfUnrKJ2m9txLJGPSyd6+jdsZeTk0NmZiYDBw4kKyuLZcuWcenSJb7//ntWrFiRL5aOHTvi6urKwIED8fLyonXr1rrXBg4cSPXq1XnhhRfYt28fly9fJjw8nAkTJnD16lXMzRQE1qvGC341CaxXTRIXYRIkeRGlwtxMwexePgD5EhgZIxcVWWkk7sUdTk1ISiMxMZHz589jaWnJhAkTWLRoEQsWLMDX15e1a9cyf/78/DEoFAwYMIATJ04wcOBAvddsbGz4448/qF27Nn379qVx48aMGDGC9PT0fD0xQpgKhVarrVCVw5KTk1GpVCQlJcl/eEYgSwQIYwkNDWXSpEm6hQ/nzJnD1q1biY6OBnKr3t67d4+tW7eW+nvn3W0EBU9ufdwckYOxtxmw8tBj32dRd3deDGpJrVq1CA0N5bnnnnuKqIUwLSW5fsucF1Gqgn3d6OrjKmPkosz169eP559/3ijvHezrxvJBLfMl7q7FTNzzem8SktILLsn//8d6oUPuMJMQlZ0kL6LU5Y2RC1GWrK2tsba2Ntr7P03injfsOnZNFArk1mQhHkfmvAghTNbPP/+Mo6MjOTk5AERHR6NQKJgxY4auzciRIxk0aBChoaF6k2GN4Wkmt+b13sityUI8nvS8CCFMVocOHbh//z7Hjx/H39+fiIgIqlevTnh4uK5NREQE06dPN16QpUiGXYUoHul5EUKYnIcXOPRu7Mve338HIDw8nLfeeovjx4+TkpLCtWvXuHjxIp06dTJyxKVHbk0W4vGk50UIYVIevWPtjrUX87/ZRNPggezbt4/58+ezceNG9u/fz507d3B3d8fb25sDBw4YOXIhRFmR5EUIYTIKWuBQWbsZt37ZzYhFm9AozGnUqBFBQUGEh4dz9+7dCtXrIoQoHhk2EkKYhMIWOLTyaII28wHJR7di5uZDjkarS17Cw8MJCgoyRrhCCCOS5EUIYRIKW+DQXGmHhbMnqWfC0br5EHn5Dh07diQqKooLFy5Iz4sQlZDBkpcPPviAtm3bYmNjU+zbF4cOHYpCodB7BAcHGypEIYQJKapEvtLDF7QalLWbkng/HScnJ3x8fHB1daVhw4ZlGKUQwhQYbHmA2bNn4+joyNWrV/nmm290JbuLMnToUG7cuMGqVat026ysrKhatWqx31eWBxCifCpuifwfRrWRIohCVEAmsTxASEgIkLveSElYWVnh6upqgIiEEKasuCXyZWVyIYTJzXkJDw/HxcWFhg0bMnbsWG7fvl1k+4yMDJKTk/UeQojyR1YmF0IUl0klL8HBwXz33Xfs2bOHBQsWEBERQffu3XWlwQsyf/58VCqV7uHh4VGGEQshSpOUyBdCFEeJ5rzMmDGDBQsWFNkmJiaGRo0a6Z4/ukx9SVy6dIl69eqxe/fuQpd+z8jIICMjQ/c8OTkZDw8PmfMiRDmWo9FKiXwhKhmDzXmZMmUKQ4cOLbJN3bp1S3LIxx6revXqXLx4sdDkxcrKCisrq1J7T1G5BAUF4efnx+LFi40diniIrExuXFlZWVhYWBg7DCEKVaJhI2dnZxo1alTkw9LSstSCu3r1Krdv38bNTbqKhWFs3ryZefPmGTsMIQwqLCyM9u3b4+joSLVq1ejZsyexsbEAxMXFoVAo2LBhA506dUKpVLJ27VoAvv76axo3boxSqaRRo0Z88cUXxjwNIXQMNuclPj6e6Oho4uPjycnJITo6mujoaFJSUnRtGjVqxJYtWwBISUnh7bff5tChQ8TFxbFnzx5eeOEF6tevT7du3QwVpqjknJycsLe3N3YYQhhUamoqkydP5ujRo+zZswczMzP69OmDRqPRtZkxYwYTJ04kJiaGbt26sXbtWmbNmsUHH3xATEwMH374Ie+99x6rV6824pkI8f+0BjJkyBAtkO/x+++/69oA2lWrVmm1Wq02LS1N+69//Uvr7OystbCw0NapU0c7atQobUJCQoneNykpSQtok5KSSvFsREXVqVMn7cSJE40dhhBl6ubNm1pAe+rUKe3ly5e1gHbx4sV6berVq6ddt26d3rZ58+ZpAwMDyzJUUYmU5PptsDovoaGhj63xon1orrC1tTU7d+40VDhCCFGpPDzpOeP2NTav/IzIyMPcunVL1+MSHx+Pj0/u7en+/v66fVNTU4mNjWXEiBGMGjVKtz07OxuVSlW2JyJEAWRVaVGpPHoXi0HKSwthZGGn1YRsP6tbK+rayjHYVqvB9P98RO92TdFoNPj6+pKZmanbx9bWVvd33vD+ypUrad26td6xzc3Ny+AMhCiaJC+i0nj0Cx3gTvxdqnqkGjEqIUpX2Gk1Y9dE6RLznAfJZN+5inXweL66aEuLNo7Y3Yst8hg1atTA3d2dS5cuMXDgQMMHLUQJSfIiKoVHv9DzZGZr2BuTSNhptRRAE+VejkZLyPazev/OzZR2mFk7cP/ETsztnJi8OAbbkxsfe6yQkBAmTJiASqUiODiYjIwMjh49yt27d5k8ebLhTkKIYjCpCrtCGEJBX+iPCtl+lhyNDCKJ8i3y8h29nkUAhcKM6v+eRmbCRa59M46L2/7LsEnvPvZYI0eO5Ouvv2bVqlU0bdqUTp06ERoaipeXl6HCF6LYpOdFVHgFfaE/Sp2UTuTlO1IYTZRrifcL/ndu7emH9cjluuc1ffz0bpjQFlJo/dVXX+XVV18t3SCFKAXS8yIqvMK+0J+0nRCmysVe+fhGJWgnhKmSnhdR4RX1Re366kfFaidEeRDg5YSbSklCUnqBw6QKche5DPByKuvQhChV0vMiKry8L/TClvVTAG7yhS4qAHMzBbN75dZtefTfe97z2b18ZJFLUe5J8iIqPPlCF5VJsK8bywe1xFWl35PoqlKyfFBLuatOVAgKbWEztcqpkiypLSqXguq8uKmUzO7lI1/oosJ5tCBjgJeTJOjCpJXk+i3Ji6hU5AtdCCFMU0mu3zJhV1Qq5mYKuR1aCCHKOZnzIoQQQohyRZIXIYQQQpQrkrwIIYQQolyR5EUIIYQQ5YokL0IIIYQoVyR5EUIIIUS5IsmLEEIIIcoVSV6EEEIIUa5I8iKEEEKIcqXCVdjNW+0gOTnZyJEIIYQQorjyrtvFWbWowiUv9+/fB8DDw8PIkQghhBCipO7fv49KpSqyTYVbmFGj0XD9+nXs7e1RKGTBvdKSnJyMh4cHf//9tyx4WYbkczcO+dyNQz534zCVz12r1XL//n3c3d0xMyt6VkuF63kxMzOjVq1axg6jwnJwcJAvFSOQz9045HM3DvncjcMUPvfH9bjkkQm7QgghhChXJHkRQgghRLkiyYsoFisrK2bPno2VlZWxQ6lU5HM3DvncjUM+d+Moj597hZuwK4QQQoiKTXpehBBCCFGuSPIihBBCiHJFkhchhBBClCuSvAghhBCiXJHkRRTbRx99hEKhYNKkScYOpcKbM2cOCoVC79GoUSNjh1XhXbt2jUGDBlGtWjWsra1p2rQpR48eNXZYFZqnp2e+f+sKhYJx48YZO7QKLScnh/feew8vLy+sra2pV68e8+bNK9a6QqagwlXYFYZx5MgRvvzyS5o1a2bsUCqNJk2asHv3bt3zKlXkP1dDunv3Lu3atePZZ5/l119/xdnZmb/++ouqVasaO7QK7ciRI+Tk5Oienz59mq5du/Lyyy8bMaqKb8GCBSxfvpzVq1fTpEkTjh49yrBhw1CpVEyYMMHY4T2WfBuKx0pJSWHgwIGsXLmS999/39jhVBpVqlTB1dXV2GFUGgsWLMDDw4NVq1bptnl5eRkxosrB2dlZ7/lHH31EvXr16NSpk5Eiqhz+/PNPXnjhBXr06AHk9oD98MMPREZGGjmy4pFhI/FY48aNo0ePHnTp0sXYoVQqf/31F+7u7tStW5eBAwcSHx9v7JAqtG3btuHv78/LL7+Mi4sLLVq0YOXKlcYOq1LJzMxkzZo1DB8+XBbWNbC2bduyZ88eLly4AMCJEyfYv38/3bt3N3JkxSM9L6JI69evJyoqiiNHjhg7lEqldevWhIaG0rBhQ9RqNSEhIXTo0IHTp09jb29v7PAqpEuXLrF8+XImT57Mf/7zH44cOcKECROwtLRkyJAhxg6vUti6dSv37t1j6NChxg6lwpsxYwbJyck0atQIc3NzcnJy+OCDDxg4cKCxQysWSV5Eof7++28mTpzIrl27UCqVxg6nUnn410+zZs1o3bo1derUYePGjYwYMcKIkVVcGo0Gf39/PvzwQwBatGjB6dOnWbFihSQvZeSbb76he/fuuLu7GzuUCm/jxo2sXbuWdevW0aRJE6Kjo5k0aRLu7u7l4t+7JC+iUMeOHSMxMZGWLVvqtuXk5PDHH3/w+eefk5GRgbm5uREjrDwcHR1p0KABFy9eNHYoFZabmxs+Pj562xo3bsymTZuMFFHlcuXKFXbv3s3mzZuNHUql8PbbbzNjxgz69+8PQNOmTbly5Qrz58+X5EWUb8899xynTp3S2zZs2DAaNWrE9OnTJXEpQykpKcTGxvLaa68ZO5QKq127dpw/f15v24ULF6hTp46RIqpcVq1ahYuLi24CqTCstLQ0zMz0p72am5uj0WiMFFHJSPIiCmVvb4+vr6/eNltbW6pVq5ZvuyhdU6dOpVevXtSpU4fr168ze/ZszM3NGTBggLFDq7Deeust2rZty4cffsgrr7xCZGQkX331FV999ZWxQ6vwNBoNq1atYsiQIVISoIz06tWLDz74gNq1a9OkSROOHz/OokWLGD58uLFDKxb5VyKECbp69SoDBgzg9u3bODs70759ew4dOpTvtlJRep555hm2bNnCzJkzmTt3Ll5eXixevLjcTGAsz3bv3k18fHy5uXBWBMuWLeO9997jjTfeIDExEXd3d15//XVmzZpl7NCKRaEtL+X0hBBCCCGQOi9CCCGEKGckeRFCCCFEuSLJixBCCCHKFUlehBBCCFGuSPIihBBCiHJFkhchhBBClCuSvAghhBCiXJHkRQghhBDliiQvQgghhChXJHkRQgghRLkiyYsQQgghyhVJXoQQQghRrvwf9i29aLZHDsoAAAAASUVORK5CYII=\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os"
      ],
      "metadata": {
        "id": "ov-chhspgDhc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# We can work with Glove or convert it to word2vec\n",
        "\n",
        "# Path 2\n",
        "from gensim.scripts.glove2word2vec import glove2word2vec\n",
        "word2vec_output_file = '/content/data/word2vec.txt'\n",
        "os.makedirs(os.path.dirname(word2vec_output_file), exist_ok=True)\n",
        "\n",
        "glove2word2vec(glove_input_file, word2vec_output_file)\n",
        "print(\"INFO: file converted and saved to - \" + word2vec_output_file)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Nzx-U_Ubau7l",
        "outputId": "d99e44f4-acca-4c37-f303-9ce548288a5c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-90-aab17ae6f751>:8: DeprecationWarning: Call to deprecated `glove2word2vec` (KeyedVectors.load_word2vec_format(.., binary=False, no_header=True) loads GLoVE text vectors.).\n",
            "  glove2word2vec(glove_input_file, word2vec_output_file)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "INFO: file converted and saved to - /content/data/word2vec.txt\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from gensim.models import KeyedVectors\n",
        "\n",
        "# load the Stanford GloVe model\n",
        "filename = 'data/word2vec.txt'\n",
        "model = KeyedVectors.load_word2vec_format(filename, binary=False)"
      ],
      "metadata": {
        "id": "ZfkkAfjogP1Y"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# calculate: (king - man) + woman = ?\n",
        "result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)\n",
        "print(result)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qZj_AQq9gY3-",
        "outputId": "98d4e154-c91a-439f-92f7-7e47918ced19"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[('queen', 0.8523604273796082)]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "# Vector operation on glove\n",
        "print(find_closest_embeddings(\n",
        "    embeddings_dict[\"woman\"] - embeddings_dict[\"man\"] + embeddings_dict[\"king\"]\n",
        ")[:5])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "morcw3xMgb51",
        "outputId": "999300e3-4afc-48d2-e0a2-00e276bc0c63"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['king', 'queen', 'prince', 'elizabeth', 'daughter']\n"
          ]
        }
      ]
    }
  ]
}