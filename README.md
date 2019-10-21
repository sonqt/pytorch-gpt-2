# pytorch-gpt-2

# Repository Structure


```
gpt2
gpt2-medium
gpt2-large
```

# Example
* **aclImdb.txt** - Dataset format

Execute the following script to tokenize your dataset

```
python3 pregenerate_training_data_gpt2.py --model_name gpt2 --train_corpus aclImdb.txt --output_dir traindata  --num_file_lines 500000 --train_batch_size 32
```

and training:
```
python3 run_gpt2.py --model_name gpt2 --do_train --do_eval --train_dataset traindata  --output_dir output_gpt --train_batch_size 32 --save_step 100 --gen_step 100
```

You can use the different pretrained model by the flag ```--model_name```. 
```
* **gpt2** - 117M
* **gpt2-medium** - 345M
* **gpt2-large** - 762M
``` 

# Sampling

```
python3 run_gpt2.py --model_type=gpt2 --length=200 --model_name_or_path=output_gpt/
```

# Results
```
i saw this film when it first came out in 1984. the snobs that were stemming quietly on my learning legs, just thought it was a study in bad guys and guns. i ended up living in japan on a train for forty five years and decided to watch it. while wearing out the last half of a billy zane dvd jacket i still had good hopes for this film when i first saw it. today i'm out looking for it on dvd. however, if i can get a copy i'll have to search until now for the never-ex discovered vhs version. the deserves a put down on a dvd collection. simply brilliant! why anyone would recommend this film to pre-teens not knows. they might dig up some dvd ideas that they may like, but they might just like it. in short - the n"e"dvd collector's delight. 10/10 for pure entertainment value. 10/10 for pure humanity.

a good film in showing how the animals can be instincts for and ventil elaborating their problems. the aspect i like the most is the contrast between the animals in the film and the wild life in the animals. it looks and sounds like the family cat and the wild life makes it very interesting. this is a great film and not a drivel.... only a accurate portrayal of the animal costumes and relations. also it shows how the wilderness is back in the 20's and 30's which includes luckily most animal talent. this is a very good film and i cannot wait to see more of it. the acting and scenery is terrific.... showing how animals can be instincts for and against each other and equally chance interactions with each other. i also appreciated the absence of violence and porn in it to keep the plot short and to let animals talk

poster says "there's no sex in this film. there's only nudity." wrong! there's no nudity in this film. there's only one nudity, and one of two sex scenes that's actually semi- intimate. overall, this is an incredibly boring film. if you're looking for sex in a plot, or characters to root for, better this than "sex in the city." there's plenty of violence and violence, but the violence is non-existent, and the plot is so dull, you may want to skip it entirely. the violence and violence might have been interesting in the comic book, but not in the film. worst of all, there's no nudity, no nudity, no violence, no nudity (other than some small male sex scenes). you can pass on this to a movie theater near you, but you will be treated to violence and violence that isn't that disgusting. i can't believe this was rated as high as it is.
```