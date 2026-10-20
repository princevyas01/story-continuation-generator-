# Story LSTM v2 vs v3 Side-by-Side Generation Comparison

## 1. Evaluation Methodology
- **Prompts**: 30 fixed prompts from `artifacts/reports/evaluation_prompts_suite.json` across 8 narrative categories.
- **Sampling Configuration**: Temperature=0.75, Top-K=40, Top-P=0.9, Repetition Penalty=1.15, Seed=42.
- **Constraint & Sanitization**: PAD, BOS, and UNK tokens banned; generation halts on EOS.

## 2. Aggregate Comparative Metrics

| Metric | Story LSTM v2 (Baseline) | Story LSTM v3 (Colab Candidate) | Difference / Notes |
| :--- | :---: | :---: | :--- |
| **Mean Latency** | 1005.1 ms | 1168.4 ms | +163.3 ms |
| **Distinct-1 (Unigram)** | 0.8917 | 0.8063 | -0.0854 |
| **Distinct-2 (Bigram)** | 0.9912 | 0.9759 | -0.0153 |
| **Repetition-4 Rate** | 0.0 | 0.0 | +0.0000 |

## 3. Detailed Side-by-Side Story Generations

### Prompt 1: [Standard Fairytale Opening]
> **Prompt**: *"Once upon a time, deep inside an ancient forest, lived a tiny squirrel who loved"*

- **v2 Output** (1345 ms):
  >  to do it.

But they said, "We are your friends. 'I will have you it," Ben asked. He was so happy and gave the park.

The little girl's mom smiled and ran around the car.

- **v3 Output** (1300 ms):
  >  to jump around. One day, while the bunny was hopping around the forest, he saw a big tree. It was very tall and the tree was very dirty. The bunny had a small idea and said he could help his friends. He quickly ran

---

### Prompt 2: [Standard Fairytale Opening]
> **Prompt**: *"Long ago in a kingdom surrounded by misty blue mountains, a young prince"*

- **v2 Output** (115 ms):
  >  of the ground.

- **v3 Output** (756 ms):
  >  and a dragon. The knight was very happy and he thanked the King for his help. He then flew away into the kingdom, never to be seen again.

---

### Prompt 3: [Standard Fairytale Opening]
> **Prompt**: *"In a peaceful village where flowers bloomed all year, there was a little house where"*

- **v2 Output** (677 ms):
  >  she had back in the lake. She was so happy to see her time, she was sad and went with them with his mom.

- **v3 Output** (604 ms):
  >  they could find out. At the end of the day, they all went to their home and their family was happy again.

---

### Prompt 4: [Standard Fairytale Opening]
> **Prompt**: *"Once upon a time, an old clockmaker found an unusual golden key inside"*

- **v2 Output** (1064 ms):
  > .

Lily to take her fun, he had lots of more mom. She was happy and they were proud of her and said!

But then, and she had a big time in the ground. The sun felt so happy to be

- **v3 Output** (1097 ms):
  > . He opened it and opened it up. Inside, there was a big, shiny and colourful key. The post had never seen it before! It was very old and had a big smile on its face. It was so beautiful, and it was

---

### Prompt 5: [Character Introduction]
> **Prompt**: *"Lily was a cheerful five-year-old girl who always wore bright yellow rain boots because"*

- **v2 Output** (1590 ms):
  >  she had all his mom and asked her.

She smiled to be so happy, but it was a bit more happy. He was so excited to give the park and said, "I are very careful!"

But she had Lily.

- **v3 Output** (1037 ms):
  >  she had found her new home. She was so happy that she made a new friend, and she hugged her mommy tightly. The end.

---

### Prompt 6: [Character Introduction]
> **Prompt**: *"Barnaby the bear was known across the woods for being clumsy, but he had the kindest"*

- **v2 Output** (1621 ms):
  >  friend. She wanted to keep her friends, and her dad saw the park. The little girl said and she didn't find a big.

But then the man was so happy and he smiled. They started to find him. He looked in

- **v3 Output** (832 ms):
  >  thing he could do. And that's what he did and it had to be careful. He would never forget this exciting adventures!

---

### Prompt 7: [Character Introduction]
> **Prompt**: *"Captain Oliver stood on the deck of his wooden sailboat and adjusted his brass telescope, wondering"*

- **v2 Output** (1488 ms):
  >  and said, "I am fun to be not so careful. You are a tree! We are scared. Can we get the dog, it is my bad of you!"

The mom smiled and started to do that she had to see the

- **v3 Output** (1528 ms):
  >  how to sail. The anchor was very happy and he kept walking until he got to a lake. He was amazed by what he saw! He decided to dive on the boat. He used the net to push the boat into the water. But the

---

### Prompt 8: [Character Introduction]
> **Prompt**: *"Zara was the most curious rabbit in the meadow, never satisfied with ordinary carrot patches when"*

- **v2 Output** (1473 ms):
  >  she had fun.

But they said, "We are my car. But the best is your friends," Ben asked. He was so happy and gave it to be very very too. She wanted to make a small room.

"

- **v3 Output** (1215 ms):
  >  you run away." The moral of the story is that sometimes you need to be careful, even if you are afraid of your own things, and that we should always take care of you.

---

### Prompt 9: [Action & Adventure]
> **Prompt**: *"The wind howled through the cave entrance as Tim grabbed his wooden staff and took"*

- **v2 Output** (1644 ms):
  >  her happy. He was scared and asked, "€ We are my old. I do we can make the story?" Lily said.

The bird says, "You't be very very prouds."

So that she said, "

- **v3 Output** (601 ms):
  >  it away. Tom and Amy were very happy and had lots of fun together.

---

### Prompt 10: [Action & Adventure]
> **Prompt**: *"Suddenly the rope bridge swung wildly from side to side over the roaring river below, and"*

- **v2 Output** (197 ms):
  >  had the ground.

- **v3 Output** (1798 ms):
  >  then he jumped back, leaving the lake and having a wonderful time. The man was so proud of himself for completing his adventure. He had been brave enough to go back to the lake, but he never forgot the wonderful times he had been in

---

### Prompt 11: [Action & Adventure]
> **Prompt**: *"The little red dragon leaped from the highest rock and flapped his tiny wings as fast as"*

- **v2 Output** (1462 ms):
  >  she had fun.

But they said, "We are my car. But the best is your friends," Ben asked. He was so happy and gave it to be very very too. She wanted to make a small room.

"

- **v3 Output** (1607 ms):
  >  he could. He was very excited to play with the missile and make it go down. When he got to the top, he saw a big tree with lots of leaves. The bird was so happy to see him, but it was too late.

---

### Prompt 12: [Action & Adventure]
> **Prompt**: *"When the mysterious trapdoor popped open in the library attic, Lucas shone his flashlight down into"*

- **v2 Output** (153 ms):
  >  the old room.

- **v3 Output** (711 ms):
  >  the sky and started to applaud. Ped was so thankful that it had found the best present ever.

---

### Prompt 13: [Dialogue Continuation]
> **Prompt**: *""Please don't go into the dark cellar," whispered Mia. But Leo smiled and replied,"*

- **v2 Output** (1586 ms):
  >  "€œBut, Lily's nice. I can eat you are you.âI will have her friend," Lily said, "I want to make your other man!"

The little girl said he went to the ground and said, "

- **v3 Output** (1711 ms):
  >  "Yes, I want to see what it is for!" Mia smiled and said, "That's right, Jake. Let's go and have some fun together." They hugged and they started to explore the woods together. They laughed and had fun,

---

### Prompt 14: [Dialogue Continuation]
> **Prompt**: *""Did you see that shining light over the lake?" asked the fox. The owl blinked slowly and said,"*

- **v2 Output** (1657 ms):
  >  "€œBut, Lily's great, I can know you have to be good, but I will want that they are so friends!

They both had a big and said, "Yes, but you't take some fun. You want

- **v3 Output** (1255 ms):
  >  "Yes, I could keep you safe. It's important to stay away from the bird." The frog smiled and thanked the bird. The bird flew away happily, feeling happy and safe.

---

### Prompt 15: [Dialogue Continuation]
> **Prompt**: *""I promise I will return before sunset," said Ella, tying her shoelaces. Her mother nodded gently and warned,"*

- **v2 Output** (1742 ms):
  >  but he felt happy. They started to her friend. She saw the other time, the bear was very happy that they started to go.

Lily thought and said, "I can you help I need to play." He's mom smiled.

- **v3 Output** (1457 ms):
  >  "It's ok, sweetheart. Let's go see if we can get a better hug." So they went off to find a new friend. They hugged each other and went back to playing in the park.

---

### Prompt 16: [Dialogue Continuation]
> **Prompt**: *""Where did all the chocolate cookies go?" shouted Dad with a suspicious frown. Sammy giggled and confessed,"*

- **v2 Output** (1781 ms):
  >  but he felt happy. They started to her friend. She saw the other time, the bear was very happy that they started to go.

Lily thought and said, "I can you help I need to play." He's mom smiled.

- **v3 Output** (1826 ms):
  >  "It's just a piece of sour ice cream. Let's try it." So, they took the ice cream into their family and the little boy's ice cream shop. The little boy was so happy that he had helped his mom. He was

---

### Prompt 17: [Moral & Lesson]
> **Prompt**: *"Even though Peter really wanted to keep the shiny coin he found on the street, he decided to"*

- **v2 Output** (184 ms):
  >  get the door.

- **v3 Output** (1720 ms):
  >  stay in his backyard. But when he looked around, he saw that the ball was so dirty. He had taken it away and was so happy to have found it. He said, "I'm sorry I didn't hurt my car. You can

---

### Prompt 18: [Moral & Lesson]
> **Prompt**: *"After laughing at his little sister when she dropped her ice cream cone, Ben realized"*

- **v2 Output** (1615 ms):
  >  her mom went for a bit and they said, "We are nice. He is a great girl."

So she was so happy. She felt angry and it was to the new friends. The mommy smiled and made his happy. They said

- **v3 Output** (638 ms):
  >  that it was an old freezer and that he could have been very sick. The end.

---

### Prompt 19: [Moral & Lesson]
> **Prompt**: *"Working together to build the treehouse was harder than doing it alone, but when the roof was finished they discovered"*

- **v2 Output** (155 ms):
  >  the old cat.

- **v3 Output** (1605 ms):
  >  that it was still a giant spot in the castle. Everyone cheered for the bird and the lion was happy to see them again. The little girl smiled and waved goodbye to the bird. She had been so brave and could fly again, and she was

---

### Prompt 20: [Moral & Lesson]
> **Prompt**: *"Greedy monkey Bob hid all the ripe mangoes for himself, but when rain came he learned that"*

- **v2 Output** (1766 ms):
  >  she had fun.

But they said, "We are my car. But the best is your friends," Ben asked. He was so happy and gave it to be very very too. She wanted to make a small room.

"

- **v3 Output** (906 ms):
  >  it was important to be honest and respectful. He learned that sometimes things can be different, like that you have done something special.

---

### Prompt 21: [Emotional Shift]
> **Prompt**: *"At first, little Emma was terrified of the dark bedroom shadows, but when she hugged her teddy bear she suddenly"*

- **v2 Output** (202 ms):
  >  of the ground.

- **v3 Output** (1313 ms):
  >  realized that she had to leave her head. She was scared and cried for help. The bunny ran away, leaving Lola alone. She was so scared that she started to cry.

---

### Prompt 22: [Emotional Shift]
> **Prompt**: *"Everyone laughed with excitement during the celebration until a sudden loud knock on the door made the room"*

- **v2 Output** (1014 ms):
  > .

Lily was scared and asked. He had never fun so happy, they said to help you. That is very happy.

- **v3 Output** (1592 ms):
  >  stop. It was a huge fire! The truck was getting closer and closer. It was going to get the oil from the truck. The little girl ran after it, but the girl was too fast. She tried to move her hands, but the

---

### Prompt 23: [Emotional Shift]
> **Prompt**: *"Toby felt heartbroken when his toy airplane snapped in half, but his grandfather smiled warmly and"*

- **v2 Output** (161 ms):
  >  asked the ground.

- **v3 Output** (526 ms):
  >  said, "It's alright. Let's go back home and explore the world."

---

### Prompt 24: [Emotional Shift]
> **Prompt**: *"The puppy was trembling under the porch after the thunderstorm, but when gentle hands offered warm milk he"*

- **v2 Output** (1526 ms):
  >  went to do the big voice. She was so happy to see her time, she said that she felt very proud of him.

"Yes, Lily, I can be so too. You are better!"

He saw a man.

- **v3 Output** (1368 ms):
  >  would get closer to the vet. He was so happy that he jumped up and down, wagging his tail. After that, the cat and the cat became best friends. They would play together and have fun together.

---

### Prompt 25: [Unseen Modern Concepts]
> **Prompt**: *"Maya sat down at her computer laptop to design a robotic helper that could sort recycled plastics, but"*

- **v2 Output** (152 ms):
  >  she had fun.

- **v3 Output** (1652 ms):
  >  she could also take a bite of it. She was very excited and asked, "What are you doing?" Mommy replied, "I'm going to teach me how to use the puzzle to make you happy." Lily was very proud of her work.

---

### Prompt 26: [Unseen Modern Concepts]
> **Prompt**: *"The solar-powered drone buzzed gently above the vegetable garden, scanning each tomato plant until"*

- **v2 Output** (1446 ms):
  >  she had back in the lake. She was so happy to see her time, she was sad and smiled. They thought that he could the mom saw it for a big boy.

- **v3 Output** (853 ms):
  >  it was gone. The hive was full of joy and laughter. And they all lived happily ever after.

---

### Prompt 27: [Unseen Modern Concepts]
> **Prompt**: *"When the Wi-Fi signal vanished in the middle of their online game, the three friends decided to"*

- **v2 Output** (194 ms):
  >  get the door.

- **v3 Output** (791 ms):
  >  stay there forever. The creature said that the world was important to keep them safe and to stay safe. The End.

---

### Prompt 28: [Rare Names & Creative Settings]
> **Prompt**: *"High upon the floating crystal islands of Zephyria, a young apprentice named Aethelgard discovered"*

- **v2 Output** (118 ms):
  >  the old room.

- **v3 Output** (618 ms):
  >  a new part of the meadow. The other animals were amazed at how much they had to keep their life as possible as they could.

---

### Prompt 29: [Rare Names & Creative Settings]
> **Prompt**: *"Deep underwater in the luminous coral dome of Thalassia, princess Nerida found a strange glowing locket that"*

- **v2 Output** (1014 ms):
  >  said, "Yes, this is too. I can know you are you. 'I will have it to come away."

"€œMom," she had to the big man. She had fun in the ground and said, "I

- **v3 Output** (1005 ms):
  >  looked like the star. Daisy was very happy and she opened her diary and saw that the message was actually a secret - she had never seen it before. It was a beautiful story! The princess was so proud of herself for finding the magical gift,

---

### Prompt 30: [Rare Names & Creative Settings]
> **Prompt**: *"In the bioluminescent hollow of Quenxar, an inquisitive badger named Vesper searched for"*

- **v2 Output** (1011 ms):
  >  her. She wanted to keep her friends.

She went around the man and the bear was happy. It would get a big, and mom she couldn't want to be so happy. She had Ben for the ground to not play with the

- **v3 Output** (1130 ms):
  >  them. Then, a mischievous bear flew out of the shadows and landed on the ground, leaving the owl on the porch. The fairy was so surprised that she ran away and told the bee that she had seen it. She told her mom what had

---

