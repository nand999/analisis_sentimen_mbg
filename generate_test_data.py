import pandas as pd
import random

random.seed(42)

# ── Pool komponen ─────────────────────────────────────────────────────────────

URLS = [
    "https://kompas.com/mbg-program-2024",
    "https://detik.com/news/mbg-berjalan",
    "https://bit.ly/mbg2024",
    "https://twitter.com/kemendikbud/status/123456",
    "https://www.instagram.com/p/mbg_hebat",
    "https://kumparan.com/berita-mbg",
    "https://liputan6.com/program-mbg",
    "https://youtu.be/mbg_review",
    "",
    "",  # banyak teks tanpa URL agar lebih natural
]

MENTIONS = [
    "@kemendikbud", "@jokowi", "@prabowo", "@nadiemmakarim",
    "@gerindra", "@dpr_ri", "@kemenkes_ri", "@makan_gratis_id",
    "@presidenri", "@setkab_ri", "@infobmkg", "@bnpb_indonesia",
    "", "", ""
]

HASHTAGS_POS = [
    "#MBG", "#MakanBergizGratis", "#GenerasiEmas", "#AnakSehat",
    "#ProgramMBG", "#BanggaIndonesia", "#GiziTerpenuhi",
    "#IndonesiaSehat", "#AnakCerdas", "#MBGMantap",
    "#DukungMBG", "#MBGBerhasil", "#MakanGratis",
]

HASHTAGS_NEG = [
    "#MBGGagal", "#HapusMBG", "#BuangAnggaran", "#KorupsiMBG",
    "#TolakMBG", "#MBGBermasalah", "#MBGNgawur",
    "#AnggaranMubazir", "#MBGTidakEfektif", "#GagalMBG",
]

EMOJIS_POS = ["😍", "🙏", "👍", "🔥", "✅", "❤️", "🥰", "💪", "🎉", "😊",
              "👏", "🌟", "⭐", "🏆", "💯", "😄", "🤩", "😁", "💚", "🫶"]
EMOJIS_NEG = ["😡", "😤", "🤮", "👎", "💀", "😒", "🤬", "😠", "❌", "🙄",
              "😑", "😞", "😔", "🤦", "🤦‍♂️", "🤦‍♀️", "💢", "😩", "😣", "🫠"]
EMOJIS_NEU = ["😐", "🤔", "🧐", "💭", "❓", "🤷", "🤷‍♂️"]

SLANG = [
    "bgt", "banget", "sgt", "gk", "gak", "ga", "udh", "udah", "emg",
    "emang", "krn", "krna", "karna", "blm", "blom", "belom", "msh",
    "masih", "jg", "juga", "tp", "tapi", "klo", "kalo", "nih", "deh",
    "sih", "dong", "yg", "yah", "wkwk", "haha", "lol", "asli", "parah",
    "gilak", "mantul", "gokil", "literally", "fr", "kuy", "gas", "kepo",
    "baper", "lebay", "bucin", "gabut", "santuy", "woles", "anjir",
    "astaga", "ya ampun", "hadeh", "aduh", "waduh", "duh", "hmm",
]

ADJ_POS = [
    "bagus", "mantap", "keren", "mantul", "top", "luar biasa",
    "bermanfaat", "berguna", "positif", "memuaskan", "oke banget",
    "the best", "super keren", "amazing", "gokil", "josss",
    "terbaik", "berkualitas", "maju", "revolusioner", "inovatif",
    "membanggakan", "mengagumkan", "spektakuler", "nggak nyangka",
]

ADJ_NEG = [
    "parah", "buruk", "jelek", "gagal", "berantakan", "gak jelas",
    "mengecewakan", "sia-sia", "zonk", "gak becus", "amburadul",
    "ngawur", "kacau", "hancur", "semrawut", "abal-abal",
    "omong kosong", "mubazir", "tidak transparan", "penuh masalah",
    "nggak guna", "asal-asalan", "alakadarnya", "setengah hati",
]

SUBJEK = [
    "gw", "gue", "aku", "saya", "kita", "kami", "adek gw",
    "anak gw", "temen gw", "guru di sekolah", "ortu murid",
    "warga sekitar", "masyarakat", "netizen", "rakyat kecil",
]

KONTEKS = [
    "di sekolah dasar", "di daerah pelosok", "di desa kami",
    "di kota besar", "di wilayah 3T", "di pinggiran kota",
    "di sekolah swasta", "di sekolah negeri", "di kampung",
    "di perkotaan", "di seluruh Indonesia",
]


# ── Template kalimat pendek/sedang ───────────────────────────────────────────

TEMPLATES_POS = [
    "program MBG {slang1} {adj_pos} {emoji_pos}",
    "makanan bergizi gratis ini {adj_pos} {slang1} {hashtag_pos}",
    "anak2 jd lebih sehat {slang1} dgn adanya MBG {emoji_pos}",
    "alhamdulillah MBG {slang1} membantu keluarga kami {hashtag_pos}",
    "{mention} program MBG keren {slang1}!! {emoji_pos} {hashtag_pos}",
    "jujur aja MBG ini {adj_pos} {slang1}, anak2 semangat sekolah {emoji_pos}",
    "gizi anak2 terpenuhi berkat MBG {slang2} {emoji_pos} {url}",
    "dukung terus program MBG {hashtag_pos} {emoji_pos} {slang1}",
    "MBG program yg {adj_pos} buat generasi penerus bangsa {emoji_pos}",
    "baca nih ttg MBG {url} {slang1} program {adj_pos} banget {hashtag_pos}",
    "anakku happy {emoji_pos} dapet makan gratis berkualitas {slang1} {hashtag_pos}",
    "salut sm pemerintah {mention} udah adain MBG {slang2} {emoji_pos}",
    "MBG {slang1} solusi nyata buat anak2 kurang mampu {emoji_pos} {hashtag_pos}",
    "cek info MBG disini {url} {emoji_pos} {slang1} mantap",
    "program ini {adj_pos} {slang1}, semoga terus berlanjut {emoji_pos} {hashtag_pos}",
    "{subjek} ngerasain langsung manfaat MBG {slang1} {emoji_pos}",
    "akhirnya ada program nyata {slang2} MBG {adj_pos} bgt {hashtag_pos}",
    "terima kasih {mention} atas program MBG yg {adj_pos} ini {emoji_pos}",
    "MBG bener2 ngubah kondisi gizi anak2 {konteks} {slang1} {emoji_pos}",
    "{subjek} liat sendiri MBG itu {adj_pos} {slang2} {emoji_pos} {hashtag_pos}",
    "gak nyangka MBG bisa sebagus ini {slang1} {emoji_pos} {url}",
    "2 jempol buat MBG {emoji_pos} {slang1} program {adj_pos} banget {hashtag_pos}",
    "mau protes tp MBG emg {adj_pos} {slang2} jadi gak bisa komplen {emoji_pos}",
    "{mention} teruskan MBG!! rakyat seneng {slang1} {emoji_pos} {hashtag_pos}",
    "MBG ini {adj_pos} {slang1} harusnya diduplikasi ke seluruh Indonesia {emoji_pos}",
    "sempet ragu tp MBG beneran {adj_pos} {slang2} {emoji_pos} {hashtag_pos}",
    "awalnya skeptis sm MBG tp ternyata {adj_pos} {slang1} {emoji_pos}",
    "laporan MBG disini {url} hasilnya {adj_pos} {slang2} {emoji_pos}",
]

TEMPLATES_NEG = [
    "MBG {slang1} {adj_neg} {emoji_neg}, buang2 duit negara",
    "kualitas makanannya {adj_neg} {slang1} {emoji_neg} {hashtag_neg}",
    "gak efektif {slang2}, anggaran habis gitu aja {emoji_neg}",
    "{mention} MBG ini {adj_neg} {slang1}!! protes nih {emoji_neg} {hashtag_neg}",
    "makan siang gratis tp gizi gak jelas {slang1} {emoji_neg} {url}",
    "parah {slang2} MBG implementasinya berantakan {emoji_neg} {hashtag_neg}",
    "program MBG sampai skrg {adj_neg} {slang1} {konteks} {emoji_neg}",
    "baca sendiri nih {url} MBG ternyata {adj_neg} {slang1} {hashtag_neg}",
    "msh banyak masalah di MBG {slang2} {emoji_neg}!! kapan beresin {mention}",
    "udh tau {slang1} MBG itu {adj_neg} tp masih dilanjutin {emoji_neg} {hashtag_neg}",
    "anggaran MBG mending dipake buat hal lain, ini {adj_neg} {slang1} {emoji_neg}",
    "{mention} tolong perbaiki MBG {slang2}!! sudah {adj_neg} {emoji_neg} {hashtag_neg}",
    "laporan MBG {url} {slang1} ternyata {adj_neg}, miris {emoji_neg}",
    "MBG cuma {adj_neg} di lapangan {slang1} {emoji_neg} {hashtag_neg}",
    "gilak {slang2} MBG gak bener {emoji_neg} {mention}",
    "{subjek} ngerasain langsung betapa {adj_neg}nya MBG {slang1} {emoji_neg}",
    "serius deh MBG itu {adj_neg} {slang2} jgn dipertahanin {emoji_neg} {hashtag_neg}",
    "sedih banget {slang1} lihat MBG yg {adj_neg} {konteks} {emoji_neg}",
    "{mention} nih buktinya MBG {adj_neg} {url} {slang2} {emoji_neg}",
    "program makan gratis kok {adj_neg} gini {slang1} {emoji_neg} {hashtag_neg}",
    "harapan besar tp realisasi MBG {adj_neg} {slang2} {emoji_neg}",
    "kecewa bgt sm MBG {slang1} {adj_neg} banget {emoji_neg} {hashtag_neg}",
    "mana janji MBG yg katanya bagus itu {slang2} kenyataannya {adj_neg} {emoji_neg}",
    "{subjek} udh capek nunggu MBG yg {adj_neg} ini diperbaiki {slang1} {emoji_neg}",
    "sampe kapan MBG tetep {adj_neg} {slang2} {mention} {emoji_neg} {hashtag_neg}",
    "percuma MBG klo implementasinya {adj_neg} {slang1} {emoji_neg} {url}",
    "udh tau dari awal MBG bakal {adj_neg} {slang2} {emoji_neg} {hashtag_neg}",
    "sayang bgt anggarannya buat MBG yg {adj_neg} {slang1} {emoji_neg}",
]


# ── Template kalimat PANJANG ──────────────────────────────────────────────────

LONG_TEMPLATES_POS = [
    (
        "Jujur nih {slang1}, {subjek} awalnya gak percaya sama program MBG ini, "
        "tapi setelah {subjek} liat langsung {konteks} ternyata emang beneran {adj_pos} {slang2}. "
        "Anak-anak keliatan lebih semangat belajar, gizinya terpenuhi, dan yang paling penting "
        "orang tua gak perlu khawatir lagi soal makan siang anak-anaknya. "
        "{mention} teruskan ya program ini!! {emoji_pos} {emoji_pos} {hashtag_pos} {hashtag_pos}"
    ),
    (
        "Sebenernya {subjek} udah lama kepo sama program MBG ini {slang1}, "
        "akhirnya kemarin bisa liat langsung {konteks}. WOW {emoji_pos} bener-bener {adj_pos} banget! "
        "Makanannya bergizi, porsinya cukup, dan yang paling mantep distribusinya teratur. "
        "Ini baru namanya program pemerintah yg beneran kerja buat rakyat {slang2}. "
        "Cek sendiri di sini {url} klo gak percaya. {hashtag_pos} {hashtag_pos} {emoji_pos}"
    ),
    (
        "{mention} mau bilang makasih {slang1} udah ngadain MBG. "
        "Program ini {adj_pos} {slang2} dan bener-bener ngebantu keluarga kurang mampu. "
        "Tadi ketemu sama ibu-ibu {konteks} yang nangis haru karena anaknya bisa makan siang gratis setiap hari. "
        "Ini bukan sekadar program, ini investasi nyata buat generasi emas Indonesia. "
        "{emoji_pos} {emoji_pos} semoga makin sukses dan diperluas ke seluruh pelosok negeri!! "
        "{hashtag_pos} {hashtag_pos} {url}"
    ),
    (
        "Thread: pengalaman {subjek} ngliat MBG {konteks} {slang1}\n"
        "1/ Program ini {adj_pos} bgt, makanannya beneran bergizi bukan asal-asalan\n"
        "2/ Anak-anak seneng {slang2}, porsinya pas dan menunya bervariasi tiap hari\n"
        "3/ Guru-guru juga bilang konsentrasi belajar anak meningkat {emoji_pos}\n"
        "4/ Harapannya semoga program ini terus berlanjut dan semakin diperluas {hashtag_pos}\n"
        "{mention} keep it up!! {emoji_pos} {emoji_pos} detail di sini {url}"
    ),
    (
        "Gak nyangka {slang1} program MBG bisa se{adj_pos} ini {emoji_pos}. "
        "{subjek} kemaren diajak temen buat survei langsung {konteks} dan hasilnya bikin salut. "
        "Proses distribusi rapi, makanan fresh, dan anak-anak pada doyan makannya. "
        "Apalagi untuk anak-anak dari keluarga kurang mampu, ini bener-bener {adj_pos} {slang2}. "
        "Negara hadir beneran nih!! {mention} pertahankan terus!! "
        "{hashtag_pos} {hashtag_pos} {emoji_pos} info lengkap: {url}"
    ),
]

LONG_TEMPLATES_NEG = [
    (
        "Serius nih {slang1}, {subjek} udah capek banget liat program MBG yg katanya {adj_pos} "
        "tapi kenyataannya {adj_neg} banget {emoji_neg}. Udah berapa bulan {konteks} tapi "
        "implementasinya masih {adj_neg} {slang2}. Makanannya gak layak, distribusinya kacau, "
        "dan anggaran entah kemana. {mention} kapan nih MBG diperbaiki?? "
        "{emoji_neg} {emoji_neg} {hashtag_neg} buktinya: {url}"
    ),
    (
        "Thread curhat MBG {slang1}: {subjek} mau cerita pengalaman pahit sama MBG {konteks}\n"
        "1/ Makanan yg dateng sering telat, kadang udah gak seger {emoji_neg}\n"
        "2/ Kualitasnya {adj_neg} {slang2}, jauh dari standar gizi yg dijanjikan\n"
        "3/ Anggaran habis besar tapi hasilnya gitu-gitu aja {emoji_neg}\n"
        "4/ Bahkan ada yg bilang makanannya bikin sakit perut {slang2} naudzubillah\n"
        "{mention} tolong dievaluasi!! ini serius {emoji_neg} {hashtag_neg} detail: {url}"
    ),
    (
        "{mention} nih mau nanya {slang1}, kenapa MBG {konteks} masih {adj_neg} banget? "
        "{subjek} udah lapor berkali-kali tapi gak ada perubahan {emoji_neg}. "
        "Anak-anak malah gak mau makan karena makanannya {adj_neg} {slang2}. "
        "Katanya program bergizi tapi kenyataannya jauh dari kata bergizi. "
        "Ini beneran mengecewakan {slang1} {emoji_neg} {emoji_neg}. "
        "Minta pertanggungjawaban!! {hashtag_neg} {hashtag_neg} {url}"
    ),
    (
        "Kocak {slang1} program MBG ini. Kata pemerintah {adj_pos}, realitanya {adj_neg} {emoji_neg}. "
        "{subjek} liat langsung {konteks} bagaimana MBG dijalankan dan {slang2} miris banget. "
        "Dana udah keluar banyak, tapi manfaatnya ke mana? Anak-anak malah gak mau makan. "
        "Ini namanya {adj_neg} tingkat dewa {emoji_neg}. Kapan sadar {mention}?? "
        "{hashtag_neg} {hashtag_neg} bukti ada di: {url}"
    ),
    (
        "FYI buat yang belum tau {slang1}, MBG {konteks} itu {adj_neg} banget {emoji_neg}. "
        "{subjek} udah survei langsung dan hasilnya bikin geleng-geleng kepala {slang2}. "
        "Makanan gak bergizi, porsi kurang, belum lagi masalah kebersihan yang {adj_neg}. "
        "Uang rakyat dipakai buat program yg {adj_neg} gini, sakit hati banget {emoji_neg}. "
        "Tolong dong {mention} evaluasi serius!! jangan janji mulu tapi {adj_neg}. "
        "{emoji_neg} {hashtag_neg} {hashtag_neg} selengkapnya: {url}"
    ),
]


# ── Helper functions ──────────────────────────────────────────────────────────

def pick(lst):
    return random.choice(lst)


def fmt(tpl):
    """Isi template dengan komponen acak."""
    return tpl.format(
        slang1=pick(SLANG),
        slang2=pick(SLANG),
        adj_pos=pick(ADJ_POS),
        adj_neg=pick(ADJ_NEG),
        emoji_pos=pick(EMOJIS_POS) if random.random() > 0.15 else "",
        emoji_neg=pick(EMOJIS_NEG) if random.random() > 0.15 else "",
        emoji_neu=pick(EMOJIS_NEU) if random.random() > 0.5 else "",
        hashtag_pos=pick(HASHTAGS_POS) if random.random() > 0.25 else "",
        hashtag_neg=pick(HASHTAGS_NEG) if random.random() > 0.25 else "",
        mention=pick(MENTIONS) if random.random() > 0.35 else "",
        url=pick(URLS) if random.random() > 0.45 else "",
        subjek=pick(SUBJEK),
        konteks=pick(KONTEKS),
    ).strip()


def build_short(n_pos, n_neg):
    texts = (
        [fmt(pick(TEMPLATES_POS)) for _ in range(n_pos)] +
        [fmt(pick(TEMPLATES_NEG)) for _ in range(n_neg)]
    )
    random.shuffle(texts)
    return texts


def build_long(n_pos, n_neg):
    texts = (
        [fmt(pick(LONG_TEMPLATES_POS)) for _ in range(n_pos)] +
        [fmt(pick(LONG_TEMPLATES_NEG)) for _ in range(n_neg)]
    )
    random.shuffle(texts)
    return texts


def build_mixed(n_pos, n_neg, long_ratio=0.2):
    """Campuran teks pendek dan panjang."""
    n_long_pos = max(1, int(n_pos * long_ratio))
    n_long_neg = max(1, int(n_neg * long_ratio))
    texts = (
        [fmt(pick(TEMPLATES_POS)) for _ in range(n_pos - n_long_pos)] +
        [fmt(pick(LONG_TEMPLATES_POS)) for _ in range(n_long_pos)] +
        [fmt(pick(TEMPLATES_NEG)) for _ in range(n_neg - n_long_neg)] +
        [fmt(pick(LONG_TEMPLATES_NEG)) for _ in range(n_long_neg)]
    )
    random.shuffle(texts)
    return texts


# ── Generate semua file CSV ───────────────────────────────────────────────────

# Test 1: Valid CSV 100 baris (campuran pendek-panjang)
texts_100 = build_mixed(50, 50, long_ratio=0.2)
pd.DataFrame({'text': texts_100}).to_csv('test_valid.csv', index=False)

# Test 2: CSV tanpa kolom text
pd.DataFrame({
    'komentar': build_short(2, 1),
    'rating': [5, 1, 3]
}).to_csv('test_no_text_column.csv', index=False)

# Test 3: CSV kosong
pd.DataFrame({'text': []}).to_csv('test_empty.csv', index=False)

# Test 4: CSV dengan tanggal
texts_20 = build_mixed(10, 10, long_ratio=0.3)
pd.DataFrame({
    'text': texts_20,
    'created_at': pd.date_range('2024-01-01', periods=20)
}).to_csv('test_with_dates.csv', index=False)

# Test 5: CSV dengan tanggal invalid
texts_10 = build_short(5, 5)
pd.DataFrame({
    'text': texts_10,
    'tanggal': ['invalid'] * 10
}).to_csv('test_invalid_dates.csv', index=False)

# Test 6: CSV 1000 baris (campuran)
texts_1000 = build_mixed(500, 500, long_ratio=0.15)
pd.DataFrame({'text': texts_1000}).to_csv('test_1000.csv', index=False)

# Test 7: CSV 10000 baris (campuran)
texts_10000 = build_mixed(5000, 5000, long_ratio=0.15)
pd.DataFrame({'text': texts_10000}).to_csv('test_10000.csv', index=False)

# # Test 8: CSV hanya positif
# texts_positive = build_mixed(100, 0, long_ratio=0.2)
# pd.DataFrame({'text': texts_positive}).to_csv('test_only_positive.csv', index=False)

# # Test 9: CSV hanya negatif
# texts_negative = build_mixed(0, 100, long_ratio=0.2)
# pd.DataFrame({'text': texts_negative}).to_csv('test_only_negative.csv', index=False)

# Test 8: CSV hanya positif
texts_positive = [
    "MBG program yang sangat bermanfaat",
    "Makanan bergizi gratis mantap sekali",
    "Program ini membantu anak-anak",
    "Sangat bagus untuk generasi emas",
] * 25
pd.DataFrame({'text': texts_positive}).to_csv('test_only_positive.csv', index=False)

# Test 9: CSV hanya negatif
texts_negative = [
    "Program ini buruk sekali",
    "Tidak efektif sama sekali",
    "Buang-buang anggaran negara",
    "Kualitas makanan jelek",
] * 25
pd.DataFrame({'text': texts_negative}).to_csv('test_only_negative.csv', index=False)

# Test 10: CSV KHUSUS teks panjang (semua dari long templates)
texts_long = build_long(50, 50)
pd.DataFrame({'text': texts_long}).to_csv('test_long_text.csv', index=False)

# Test 11: CSV teks panjang + tanggal
pd.DataFrame({
    'text': texts_long[:40],
    'created_at': pd.date_range('2023-06-01', periods=40, freq='3D')
}).to_csv('test_long_with_dates.csv', index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("✅ Semua file test CSV berhasil dibuat!")
print("\nFile yang dibuat:")
print("1.  test_valid.csv             (100 baris — campuran pendek+panjang)")
print("2.  test_no_text_column.csv")
print("3.  test_empty.csv")
print("4.  test_with_dates.csv        (20 baris + kolom created_at)")
print("5.  test_invalid_dates.csv     (10 baris + tanggal invalid)")
print("6.  test_1000.csv              (1000 baris — campuran)")
print("7.  test_10000.csv             (10000 baris — campuran)")
print("8.  test_only_positive.csv     (100 baris positif)")
print("9.  test_only_negative.csv     (100 baris negatif)")
print("10. test_long_text.csv         (100 baris TEKS PANJANG)")
print("11. test_long_with_dates.csv   (40 baris teks panjang + tanggal)")

print("\n── Contoh teks PENDEK ──")
short_samples = build_short(3, 2)
for t in short_samples:
    print(" •", t)

print("\n── Contoh teks PANJANG ──")
long_samples = build_long(2, 1)
for t in long_samples:
    print(" •", t[:200], "..." if len(t) > 200 else "")
    print()
