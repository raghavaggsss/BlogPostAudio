**"Playing" with Audio**
================

Audio processing and comprehension is a very interesting and increasingly popular as well as useful area of study. A lot of methods have been devised and even more are being studied currently. Its applications are wide such as Youtube auto captions, PlayMusic lyrics, Shazam, Smule, Siri, Cortana, Google Speech to Text and many more. The purpose of this blog is to get a newbie acquainted to some of the very basic procedures to extract features from an audio recording or a sound track.
</n> 
</n>
</n>

>*You can read about one of the most fundamental audio signal processing problem **"The cocktail party problem" ** here-*
 [The Cocktail Party Problem](http://www.brainfacts.org/sensing-thinking-behaving/awareness-and-attention/articles/2013/the-cocktail-party-problem/) <\n><\n>
 >*An interesting solution to the same can be found here -*
 [Deep learning to solve Cocktail Party Problem](https://www.technologyreview.com/s/537101/deep-learning-machine-solves-the-cocktail-party-problem/)  *(Yes deep learning is magical!)*
 






Basic Audio Features
-------------

First, we'll talk about some basic features of an audio file. Intuitively, features like **frequency, power, amplitude, phase, zero crossings** etc sound relevant. Yes, they are indeed important but we often use features which are simple or complicated functions of these simple features.



- One such suite of features is the **MFCCs**. MFCCs analysis is focused on extracting features for automatic speech recognition systems. This analysis includes filtering linguistic features and discarding background noise, emotion etc. Details can be found [here](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) .

- Extracting musical features of often more complicated and involves dealing directly with an **audio spectrogram**.  Below is a typical spectrogram of an audio track plotted using [Sonic Visualiser](http://www.sonicvisualiser.org/).

![enter image description here](https://doc-0k-5k-docs.googleusercontent.com/docs/securesc/kjgkmthj3i3paitkpd8rd3s5ilks3s7d/c41k2ul5i6c9qmagcccgf47chdci27ef/1502964000000/05786851616636697596/05786851616636697596/0B-adnOtqAB4AYUh6WWRJaUZOUDA?e=download&nonce=799hlpi27tu2q&user=05786851616636697596&hash=665o3hrg99315d69f5v7tb074e6hqnn5)




> **Spectrogram is a plot of amount of frequencies contained in a wave vs the time. **
> 
> - The frequency regions which seem to be **Hot** *(in the sense that they are more luminous)* have a greater intensity fraction.

> - A simple application of the spectrogram includes finding the dominant pitch in a certain time frame hence helping find the key of song. It can also be used to analyse different voices in an audio sample.





Fast Fourier Transformation (FFT)
----------------------------------------------

To analyse a complex wave, we break it down into various simple sinusoidal waves with different frequencies such that the superimposition of these simple waves give back the original wave. 
Following is a **Discrete Fourier Transform (DFT)** of a signal x(t<sub>n</sub>).
<center>![DFT Equation](https://doc-0c-5k-docs.googleusercontent.com/docs/securesc/kjgkmthj3i3paitkpd8rd3s5ilks3s7d/jo6mbkr1a06a4h5oov4hd4b2gagnt6kp/1502971200000/05786851616636697596/05786851616636697596/0B-adnOtqAB4AQlhObDktdmlxU0U?e=download)</center>

<center>![Symbols](https://doc-04-5k-docs.googleusercontent.com/docs/securesc/kjgkmthj3i3paitkpd8rd3s5ilks3s7d/t1d8bsqbgd0hscquo81v9rfks2o6g478/1502971200000/05786851616636697596/05786851616636697596/0B-adnOtqAB4AdEluWjREdFBPMkE?e=download)</center>

FFT is an efficient algorithm to calculate the DFT of a wave. DFT takes **N<sup>2</sup>** operations whereas FFT takes only **Nlog<sub>2</sub>N** operations.
</n>

----------------------
<h4>FFT Implementation in Python using <b>Scipy</b></h4>

Scipy is an open source numerical and scientific computation library for python which comes with *fft* package. Following is an implementation of the *fft* function defined in the scipy *fftpack*. 
**Input signal** - *x(N) = sin(50(2πNT)) + 0.5sin(80(2πNT))*
```python
>>> from scipy.fftpack import fft
>>> # Number of sample points
>>> N = 600
>>> # sample spacing
>>> T = 1.0 / 800.0
>>> x = np.linspace(0.0, N*T, N)
>>> y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
>>> yf = fft(y)
>>> xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
>>> import matplotlib.pyplot as plt
>>> plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
>>> plt.grid()
>>> plt.show()
```
<center>
![FFT Plot](https://doc-0k-5k-docs.googleusercontent.com/docs/securesc/kjgkmthj3i3paitkpd8rd3s5ilks3s7d/n0oa6nqaq7a6ro5nv33273bkeoue5mm8/1502971200000/05786851616636697596/05786851616636697596/0B-adnOtqAB4ANHBRSzRFNGozVWM?e=download)
</center>

- Clearly the graph shows the distribution of the signal in various frequencies and matches our intuition because the input wave is essentially a combination of two simple waves of frequency 50Hz and 80Hz each.

--------------
<h4>Implementing on a wav file</h4>

```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from scipy.io.wavfile import read
>>> #read sampling rate and signal values from the file
>>> rate, signal = read('file.wav','r')
>>> time = np.arange(len(signal))
>>> time = time/float(rate)
>>> #plot a section of the audio file
>>> plt.plot(time[7000:10000],signal[7000:10000])
>>> plt.show()

```


<center>
![Plotting wave](https://doc-14-5k-docs.googleusercontent.com/docs/securesc/kjgkmthj3i3paitkpd8rd3s5ilks3s7d/k2mq0q9aa79qj8ladl72cu43hb3uki3d/1502978400000/05786851616636697596/05786851616636697596/0B-adnOtqAB4AUjBXb2N0Z2NZQWc?e=download)
</center>

<above steps link to FFT using scipy>
Treating this as the input wave, the above steps can be repeated to get the FFT of any audio file.




#### <i class="icon-refresh"></i> Open a document

You can open a document from <i class="icon-provider-gdrive"></i> **Google Drive** or the <i class="icon-provider-dropbox"></i> **Dropbox** by opening the <i class="icon-refresh"></i> **Synchronize** sub-menu and by clicking **Open from...**. Once opened, any modification in your document will be automatically synchronized with the file in your **Google Drive** / **Dropbox** account.

#### <i class="icon-refresh"></i> Save a document

You can save any document by opening the <i class="icon-refresh"></i> **Synchronize** sub-menu and by clicking **Save on...**. Even if your document is already synchronized with **Google Drive** or **Dropbox**, you can export it to a another location. StackEdit can synchronize one document with multiple locations and accounts.

#### <i class="icon-refresh"></i> Synchronize a document

Once your document is linked to a <i class="icon-provider-gdrive"></i> **Google Drive** or a <i class="icon-provider-dropbox"></i> **Dropbox** file, StackEdit will periodically (every 3 minutes) synchronize it by downloading/uploading any modification. A merge will be performed if necessary and conflicts will be detected.

If you just have modified your document and you want to force the synchronization, click the <i class="icon-refresh"></i> button in the navigation bar.

> **Note:** The <i class="icon-refresh"></i> button is disabled when you have no document to synchronize.

#### <i class="icon-refresh"></i> Manage document synchronization

Since one document can be synchronized with multiple locations, you can list and manage synchronized locations by clicking <i class="icon-refresh"></i> **Manage synchronization** in the <i class="icon-refresh"></i> **Synchronize** sub-menu. This will let you remove synchronization locations that are associated to your document.

> **Note:** If you delete the file from **Google Drive** or from **Dropbox**, the document will no longer be synchronized with that location.

----------


Publication
-------------

Once you are happy with your document, you can publish it on different websites directly from StackEdit. As for now, StackEdit can publish on **Blogger**, **Dropbox**, **Gist**, **GitHub**, **Google Drive**, **Tumblr**, **WordPress** and on any SSH server.

#### <i class="icon-upload"></i> Publish a document

You can publish your document by opening the <i class="icon-upload"></i> **Publish** sub-menu and by choosing a website. In the dialog box, you can choose the publication format:

- Markdown, to publish the Markdown text on a website that can interpret it (**GitHub** for instance),
- HTML, to publish the document converted into HTML (on a blog for example),
- Template, to have a full control of the output.

> **Note:** The default template is a simple webpage wrapping your document in HTML format. You can customize it in the **Advanced** tab of the <i class="icon-cog"></i> **Settings** dialog.

#### <i class="icon-upload"></i> Update a publication

After publishing, StackEdit will keep your document linked to that publication which makes it easy for you to update it. Once you have modified your document and you want to update your publication, click on the <i class="icon-upload"></i> button in the navigation bar.

> **Note:** The <i class="icon-upload"></i> button is disabled when your document has not been published yet.

#### <i class="icon-upload"></i> Manage document publication

Since one document can be published on multiple locations, you can list and manage publish locations by clicking <i class="icon-upload"></i> **Manage publication** in the <i class="icon-provider-stackedit"></i> menu panel. This will let you remove publication locations that are associated to your document.

> **Note:** If the file has been removed from the website or the blog, the document will no longer be published on that location.

----------


Markdown Extra
--------------------

StackEdit supports **Markdown Extra**, which extends **Markdown** syntax with some nice features.

> **Tip:** You can disable any **Markdown Extra** feature in the **Extensions** tab of the <i class="icon-cog"></i> **Settings** dialog.

> **Note:** You can find more information about **Markdown** syntax [here][2] and **Markdown Extra** extension [here][3].


### Tables

**Markdown Extra** has a special syntax for tables:

Item     | Value
-------- | ---
Computer | $1600
Phone    | $12
Pipe     | $1

You can specify column alignment with one or two colons:

| Item     | Value | Qty   |
| :------- | ----: | :---: |
| Computer | $1600 |  5    |
| Phone    | $12   |  12   |
| Pipe     | $1    |  234  |


### Definition Lists

**Markdown Extra** has a special syntax for definition lists too:

Term 1
Term 2
:   Definition A
:   Definition B

Term 3

:   Definition C

:   Definition D

	> part of definition D


### Fenced code blocks

GitHub's fenced code blocks are also supported with **Highlight.js** syntax highlighting:

```
// Foo
var bar = 0;
```

> **Tip:** To use **Prettify** instead of **Highlight.js**, just configure the **Markdown Extra** extension in the <i class="icon-cog"></i> **Settings** dialog.

> **Note:** You can find more information:

> - about **Prettify** syntax highlighting [here][5],
> - about **Highlight.js** syntax highlighting [here][6].


### Footnotes

You can create footnotes like this[^footnote].

  [^footnote]: Here is the *text* of the **footnote**.


### SmartyPants

SmartyPants converts ASCII punctuation characters into "smart" typographic punctuation HTML entities. For example:

|                  | ASCII                        | HTML              |
 ----------------- | ---------------------------- | ------------------
| Single backticks | `'Isn't this fun?'`            | 'Isn't this fun?' |
| Quotes           | `"Isn't this fun?"`            | "Isn't this fun?" |
| Dashes           | `-- is en-dash, --- is em-dash` | -- is en-dash, --- is em-dash |


### Table of contents

You can insert a table of contents using the marker `[TOC]`:

[TOC]


### MathJax

You can render *LaTeX* mathematical expressions using **MathJax**, as on [math.stackexchange.com][1]:

The *Gamma function* satisfying $\Gamma(n) = (n-1)!\quad\forall n\in\mathbb N$ is via the Euler integral

$$
\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.
$$

> **Tip:** To make sure mathematical expressions are rendered properly on your website, include **MathJax** into your template:

```

```

> **Note:** You can find more information about **LaTeX** mathematical expressions [here][4].


### UML diagrams

You can also render sequence diagrams like this:

```sequence
Alice->Bob: Hello Bob, how are you?
Note right of Bob: Bob thinks
Bob-->Alice: I am good thanks!
```

And flow charts like this:

```flow
st=>start: Start
e=>end
op=>operation: My Operation
cond=>condition: Yes or No?

st->op->cond
cond(yes)->e
cond(no)->op
```

> **Note:** You can find more information:

> - about **Sequence diagrams** syntax [here][7],
> - about **Flow charts** syntax [here][8].

### Support StackEdit

[![](https://cdn.monetizejs.com/resources/button-32.png)](https://monetizejs.com/authorize?client_id=ESTHdCYOi18iLhhO&summary=true)

  [^stackedit]: [StackEdit](https://stackedit.io/) is a full-featured, open-source Markdown editor based on PageDown, the Markdown library used by Stack Overflow and the other Stack Exchange sites.


  [1]: http://math.stackexchange.com/
  [2]: http://daringfireball.net/projects/markdown/syntax "Markdown"
  [3]: https://github.com/jmcmanus/pagedown-extra "Pagedown Extra"
  [4]: http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
  [5]: https://code.google.com/p/google-code-prettify/
  [6]: http://highlightjs.org/
  [7]: http://bramp.github.io/js-sequence-diagrams/
  [8]: http://adrai.github.io/flowchart.js/
