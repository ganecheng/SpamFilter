package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Test
{

	public static void main(String[] args)
	{
		// 得到所有单词
		String[] banword = getAllWordsFromFile("TrainingSet/SMSSpamCollection");

		// 预热数据,将所有关键字放在一个Map中
		Map<String, KeywordCount> keywordMap = new HashMap<String, KeywordCount>();
		for (String s : banword)
		{
			keywordMap.put(s, new KeywordCount(s, 0, 0, 0, 0));
		}

		// 得到所有训练邮件列表
		List<String> mailList = getContentFromFile("TrainingSet/SMSSpamCollection");

		// 统计垃圾邮件出现的次数
		int spamNumber = 0;
		// 统计正常邮件出现的次数
		int legitNumber = 0;
		// 统计每个关键字在正常邮件和垃圾邮件中出现的次数
		for (int i = 0; i < mailList.size(); i++)
		{
			String mailContent = mailList.get(i);

			// 看真实情况是否是垃圾邮件
			if (mailContent.startsWith("spam"))
			{
				for (Map.Entry<String, KeywordCount> entry : keywordMap.entrySet())
				{
					boolean containFlag = FilterKeyWord(mailContent, entry.getKey());
					KeywordCount keywordCount = entry.getValue();

					if (containFlag == true)
					{
						keywordCount.spam += 1;
					}
					keywordCount.spamAll += 1;
				}
				spamNumber++;
			}
			else
			{
				for (Map.Entry<String, KeywordCount> entry : keywordMap.entrySet())
				{
					boolean containFlag = FilterKeyWord(mailContent, entry.getKey());
					KeywordCount keywordCount = entry.getValue();

					if (containFlag == true)
					{
						keywordCount.legit += 1;
					}
					keywordCount.legitAll += 1;

				}
				legitNumber++;
			}
		}

		List<String> needRemoveKey = new ArrayList<String>();
		// 得到每一个关键字出现的情况下是垃圾邮件的概率的概率
		for (Map.Entry<String, KeywordCount> entry : keywordMap.entrySet())
		{
			KeywordCount kcTemp = entry.getValue();
			if (kcTemp.spam + kcTemp.legit == 0)
			{
				needRemoveKey.add(entry.getKey());
				continue;
			}

			double Spam = 1.0 * kcTemp.spam / kcTemp.spamAll;
			double SpamAll = 1.0 * kcTemp.spamAll / (kcTemp.spamAll + kcTemp.legitAll);
			double Legit = 1.0 * kcTemp.legit / kcTemp.legitAll;
			double LegitAll = 1.0 * kcTemp.legitAll / (kcTemp.spamAll + kcTemp.legitAll);

			kcTemp.combiningProbabilities = (Spam * SpamAll) / (Spam * SpamAll + Legit * LegitAll); // 根据（公式2）

			if (kcTemp.combiningProbabilities < 0.93)
			{
				needRemoveKey.add(entry.getKey());
			}
		}

		// 过滤得到所有符合要求的对垃圾邮件有较高识别度的关键词
		for (String s : needRemoveKey)
		{
			keywordMap.remove(s);
		}

		// 查看结果
		for (Map.Entry<String, KeywordCount> entry : keywordMap.entrySet())
		{
			System.out.println(entry.getValue());
		}

		// 得到所有测试邮件列表
		List<String> testMailList = getContentFromFile("TestSet/TestFile.txt");
		// 成功识别的数量
		int rightCount = 0;
		//识别错误的数量
		int wrongCount = 0;
		// 总共垃圾邮件数量
		int spamCount = 0;
		for (String mail : testMailList)
		{
			// 找到这封邮件含有的关键字
			String thisMail = mail;

			// 总共垃圾邮件数量
			if (thisMail.startsWith("spam"))
			{
				spamCount++;
			}

			List<String> oneMailKeywordList = new ArrayList<String>();

			for (Map.Entry<String, KeywordCount> entry : keywordMap.entrySet())
			{
				boolean containFlag = FilterKeyWord(thisMail, entry.getKey());
				if (containFlag == true)
				{
					oneMailKeywordList.add(entry.getKey());
				}
			}

			if (oneMailKeywordList.size() <= 0)
			{
				// System.out.println("没有含有关键字,应该是正常邮件");
				continue;
			}

			// 得到这封邮件所有关键词的联合概率,根据(公式3)
			double Pup = 1.0 * spamNumber / (spamNumber + legitNumber);
			double Pdown = 1.0f;
			for (String kw : oneMailKeywordList)
			{
				Pup = Pup * keywordMap.get(kw).spam / keywordMap.get(kw).spamAll;
				Pdown = Pdown * (keywordMap.get(kw).spam + keywordMap.get(kw).legit) / (spamNumber + legitNumber);
			}
			double Pmail = Pup / (Pup + Pdown);

			System.out.println("该封邮件是垃圾邮件的概率为:" + Pmail + ",实际是否为垃圾邮件:" + thisMail.startsWith("spam"));

			// 成功识别
			if (Pmail > 0.999 && thisMail.startsWith("spam"))
			{
				rightCount++;
			}
			// 识别错误
			if (Pmail > 0.999 && thisMail.startsWith("ham"))
			{
				wrongCount++;
			}
		}
		System.out.println("垃圾邮件总数为" + spamCount + ",正确识别了" + rightCount + "封垃圾邮件，召回率" + rightCount * 1.0 / spamCount + ",准确率：" + rightCount * 1.0
				/ (rightCount + wrongCount));
	}

	/**
	 * 将banword 的关键字词与邮件内容逐字比较，若邮件内容中包含此关键字，则返回true
	 * 
	 * @param strContent
	 * @param strKeyWord
	 * @return
	 */
	private static boolean FilterKeyWord(String strContent, String strKeyWord)
	{
		boolean retVal = false;

		if (strContent.toLowerCase().indexOf(strKeyWord.toLowerCase()) >= 0)
		{
			retVal = true;
		}

		return retVal;
	}

	/**
	 * 读取文件内容
	 * 
	 * @param fileName
	 * @return
	 */
	public static List<String> getContentFromFile(String fileName)
	{
		List<String> totalList = new ArrayList<String>();
		try
		{
			File file = new File(fileName);

			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
			String str;
			while ((str = br.readLine()) != null)
			{
				totalList.add(str.trim());
			}
			br.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}

		return totalList;
	}

	/**
	 * 从文件中获得所有的单词
	 * 
	 * @param fileName
	 * @return
	 */
	public static String[] getAllWordsFromFile(String fileName)
	{
		StringBuffer sb = new StringBuffer();
		try
		{
			File file = new File(fileName);

			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
			String str;
			while ((str = br.readLine()) != null)
			{
				sb.append(str.replaceAll("spam", "").replaceAll("ham", "").trim());
			}
			br.close();
		}
		catch (IOException e)
		{
			e.printStackTrace();
		}

		String[] a = sb.toString().split("[^a-zA-Z]+");
		return a;
	}
}

class KeywordCount
{

	public KeywordCount(String keyword, int spam, int spamAll, int legit, int legitAll)
	{
		super();
		this.keyword = keyword;
		this.spam = spam;
		this.spamAll = spamAll;
		this.legit = legit;
		this.legitAll = legitAll;
	}

	// 关键词
	public String keyword;
	// 此关键词在垃圾邮件中出现的次数
	public int spam;
	// 垃圾邮件总数量
	public int spamAll;
	// 此关键词在正常邮件中出现的次数
	public int legit;
	// 正常邮件总数量
	public int legitAll;
	// 这个关键词存在的情况下,是垃圾邮件的概率
	public double combiningProbabilities;

	@Override
	public String toString()
	{
		return "[keyword=" + keyword + ", spam=" + spam + ", spamAll=" + spamAll + ", legit=" + legit + ", legitAll=" + legitAll + ", combiningProbabilities="
				+ combiningProbabilities + "]";
	}

}
