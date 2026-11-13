Lokalni ParlaSpeech plugin za prepoznavanje i pretvorbu hrvatskog govora u tekst.

Za razliku od Whispera, ovaj je model specijaliziran za hrvatski jezik. Dodatno je treniran s 300 sati snimki iz hrvatskog parlamenta u sklopu skupa podataka [ParlaSpeech-HR v1.0](http://hdl.handle.net/11356/1494).

Sve se odvija lokalno, bez API ključeva, usluga u oblaku ili vanjskih poslužitelja.

**Obavezno uključi DSH Voice Input plugin** (`@deepseek-ai/dsh-experimental-voice-input-bundle`)


## Glavne značajke

- **Optimiziran za hrvatski jezik:** WER od 7,61 % na testnom skupu ParlaSpeech-HR (za Whisper small navodi se približno 15 - 20 %).
- **Višejezična podrška:** Podržava hrvatski (`hr`) i automatsko prepoznavanje jezika (`auto`).
- **Rad bez interneta:** Ne zahtijeva API ključ ni internetsku vezu nakon jednokratnog preuzimanja modela.
- **Bez zasebnog poslužitelja:** Koristi postojeće mogućnosti DSH Voice Input dodatka (mikrofon, valni prikaz i umetanje teksta u poruku).
- **Brza transkripcija:** Koristi trajni Python pozadinski proces (*worker*), pa se model učitava samo jednom.
- **GPU ubrzanje:** Automatski koristi CUDA tehnologiju ako je dostupna.
- **Potpuna privatnost:** Radi isključivo **lokalno** na tvom računalu.


## Modeli

Modeli su dostupni na [Hugging Faceu](https://huggingface.co/classla).

| model | disk | RAM | WER | napomena |
|-------|------|-----|-----|----------|
| `classla/wav2vec2-xls-r-parlaspeech-hr` | ~1.2 GB | ~2 GB | 7.61% | zadano — dobar omjer kvalitete i brzine |
| `classla/wav2vec2-large-slavic-parlaspeech-hr-lm` | ~3.6 GB | ~4 GB | ~5% | najbolja točnost, zahtijeva više resursa |

> **Napomena:** Plugin je namjenski napravljen za hrvatski jezik. Iako se
> `modelId` može postaviti na bilo koji `Wav2Vec2ForCTC` model
> (uključujući modele za druge jezike), službeno je testiran i podržan samo
> hrvatski jezik.


• Prilikom **prve upotrebe** plugin stvori Python okruženje u `runtime/venv/`, instalira Python pakete i preuzme model s Hugging Facea. Zadani model zauzima približno 1,2 GB, a pohranjuje se u `~/.cache/huggingface/`.

## Preduvjet

- Originalni eksperimentalni DSH Voice Input plugin mora biti uključen
- deepseek-harness `>= 0.2.2-alpha.2`
- **Python 3.9+** s paketima: `transformers`, `torch`, `soundfile`, `librosa`, `numpy`
- ~1.2–3.6 GB slobodnog diska za model (ovisno o odabranom modelu)
- Pri prvom pokretanju potrebna je internetska veza za preuzimanje modela s Hugging Facea. Nakon toga plugin radi potpuno offline.

## Postavke

Plugin se može koristiti na **dva načina**, ovisno o tome je li Python okruženje već pripremljeno. Oba se postavljaju u datoteci `cordis.patch.yml`.

### Način 1 — Napredni korisnik (sve je već instalirano)

Ako već imaš Python okruženje s paketima `transformers`, `torch`, `soundfile`, `librosa`, `numpy` i lokalnu kopiju modela, usmjeri plugin na njih. Plugin **ne kreira venv, ne instalira pakete i ne preuzima model**

```yaml
- insert:
    - id: speech-to-text-parlaspeech-hr
      name: dsh-voice-parlaspeech-hr
      config:
        providerId: parlaspeech-hr
        modelId: classla/wav2vec2-xls-r-parlaspeech-hr
        modelDirectory: /putanja/do/modela
        pythonPath: /putanja/do/venv/bin/python
```      
### Način 2 — Novi korisnik (ništa nije instalirano)

Ostavi `modelDirectory` i `pythonPath` kao prazne vrijednosti (`''`) ili ih u potpunosti izostavi. 
Plugin tada pri prvom korištenju automatski:
- kreira virtualno okruženje u `runtime/venv/`,
- instalira Python pakete (`transformers`, `torch`, `soundfile`, `librosa`, `numpy`),
- preuzima model s Hugging Facea u `~/.cache/huggingface/`.

```yaml
- insert:
    - id: speech-to-text-parlaspeech-hr
      name: dsh-voice-parlaspeech-hr
      config:
        providerId: parlaspeech-hr
        modelId: classla/wav2vec2-xls-r-parlaspeech-hr
        modelDirectory: ''
        pythonPath: ''
```        
     
### Instalacija potrebnih Python paketa

```bash
pip install transformers torch soundfile librosa numpy
