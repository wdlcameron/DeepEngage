from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import random
import urllib
import numpy as np
from pathlib import Path
import pandas as pd
from functools import partial
import re
import datetime
import imageio


class InstagramBot():
    """
    Bot used to scrape information from instagram and perform common features including liking posts
    
    To avoid getting your account banned during troubleshooting, make sure to reinitialize
    your bot with your current browser rather than opening up a new one each time.
    
    
    
    """
    def __init__(self, email = None, password = None, browser = None):
        self.browser = webdriver.Chrome() if browser is None else browser
        self.email = email
        self.password = password
        
        
    def signIn(self, email = None, password = None):
        """
        This will sign into instagram using either the provided username and password
        or the ones that are stored in the bot
        """

        self.email = self.email if email is None else email
        self.password = self.password if password is None else password

        assert self.email is not None and self.password is not None, "No username or password available"

        self.browser.get('https://www.instagram.com/accounts/login/')
        time.sleep(2)
        emailInput = self.browser.find_elements_by_css_selector('form input')[0]
        passwordInput = self.browser.find_elements_by_css_selector('form input')[1]
        
        emailInput.send_keys(self.email)
        passwordInput.send_keys(self.password)
        passwordInput.send_keys(Keys.ENTER)
        time.sleep(2)
        return True
        
    def followerList(self, user, max_followers = 1000):
        self.browser.get('https://www.instagram.com/' + user)
        
        time.sleep(20)
        allLinks = self.browser.find_elements_by_css_selector('ul li a')

        followLink = None
        for link in allLinks:
            if link.text.endswith('followers'):
                followLink = link
                break

        followLink.click()
        time.sleep(1)
        followersList = self.browser.find_element_by_css_selector('div[role=\'dialog\'] ul')
        numberFollowersList = len(followersList.find_elements_by_css_selector('li'))
        lastFollowersList = 0
        followersList.find_element_by_css_selector('li').click()

        actionChain = webdriver.ActionChains(self.browser)
        while (numberFollowersList< max_followers and lastFollowersList != numberFollowersList):
            followersList.find_elements_by_css_selector('li')[-1].click()
            lastFollowersList = numberFollowersList
            actionChain.key_down(Keys.SPACE).key_up(Keys.SPACE).perform()
            time.sleep(1)
            numberFollowersList = len(followersList.find_elements_by_css_selector('li'))
            #print(numberFollowersList, lastFollowersList)

        followers = []
        for user in followersList.find_elements_by_css_selector('li'):
            userLink = user.find_element_by_css_selector('a').get_attribute('href')
            followers.append(userLink)
            
        return followers
    
    def goto(self, url):
        self.browser.get(url)
        
        
    def check_private(self):
        return bool(self.browser.find_elements_by_xpath("//*[contains(text(), 'This Account is Private')]"))
    
    
    def check_public(self):
        return not self.check_private()
    
    
    def like_post(self):
        like_button = self.browser.find_elements_by_css_selector('span[aria-label=\'Like\']')
        if bool(like_button): 
            like_button[0].click()
            return True
        else:
            return False
            
    def next_post(self):
        next_button = self.browser.find_elements_by_xpath("//*[contains(text(), 'Next')]")
        if bool(next_button): 
            next_button[0].click()
            return True
        else: return False
        
    def first_post(self):
        first_pic = self.browser.find_elements_by_class_name("_9AhH0")
        if bool(first_pic): 
            first_pic[0].click()
            return True
        else: return False
        
        
    def wait_random(self, low = 1, high = 10):
        wait_time = random.randint(low, high)
        time.sleep(wait_time/10)
        
    def check_action_block(self):
        block_button = self.browser.find_elements_by_xpath("//*[contains(text(), 'This action was blocked')]")
        return bool(block_button)
    
    def reset_activity(self):
        active_button = self.browser.find_elements_by_css_selector('svg[aria-label=\'Activity Feed\']')
        try:
            if bool(active_button): 
                active_button[0].click()
                
                self.wait_random(40,60)
                
                close_activity_button = self.browser.find_elements_by_class_name('_8Mwnh')
                close_activity_button[0].click()
                
                return True
        
            else:
                return False
        except Exception as e:
            print (e)
            return False
        
        
        
    def renavigation_check(self, url):
        if url is not None and not (url == self.browser.current_url): 
            self.browser.get(url)
            time.sleep(1)
        url = self.browser.current_url if url is None else url
        return url


    
    def like_posts_by_user(self, user, num_posts = 3):
        self.browser.get(user)
        try:

            
            self.wait_random(10,30)
            assert self.check_public(), 'Private account'
            self.wait_random(9,15)
            #Check if there is any activity thaqt needs to be cleared (There is a hidden container that prevents clicking)
            if len(self.browser.find_elements_by_class_name("H9zXO")): 
                assert self.reset_activity(), "Problem resetting activity"
                self.wait_random(10, 30)
            assert self.first_post(), 'No first post'
            self.wait_random(15,30)
            assert self.like_post(), 'Already liked'
            self.wait_random(9,20)
            
            for i in range(num_posts-1):
                for _ in range(random.randint(1,3)):
                    assert self.next_post(), 'Last post'
                    self.wait_random(10,40)
                assert self.like_post(), 'Already liked'
                self.wait_random(9,20)
            
            
            actionChain = webdriver.ActionChains(self.browser)
            actionChain.key_down(Keys.ESCAPE).key_up(Keys.ESCAPE).perform()
            
            #self.wait_random(10, 30)
            #assert self.reset_activity(), "Problem resetting activity"
            #self.wait_random(30, 80)
            
        except AssertionError as e:
            print (e)
            if self.check_action_block(): 
                print('Action Blocked!!')
                return 'Block'
            return False
        except Exception as e:
            print('Unknown error', e)
            if self.check_action_block(): 
                print('Action Blocked!!')
                return 'Block'
        finally:
            actionChain = webdriver.ActionChains(self.browser)
            actionChain.key_down(Keys.ESCAPE).key_up(Keys.ESCAPE).perform()
        
        return True
    
    
    """
    Extract a single metric
    """
        
    def get_username(self): return self.browser.find_element_by_class_name("nJAzx").text
        
    def get_likes(self):
        likes_span = self.browser.find_element_by_class_name('Nm9Fw')
        likes = int(likes_span.find_element_by_tag_name('span').text.replace(',', ''))
        return likes
    
    def get_post_time(self): return self.browser.find_element_by_class_name('_1o9PC').get_attribute('datetime')
    
    def get_alt_text(self): return self.browser.find_element_by_class_name('FFVAD').get_attribute('alt')
    
    def get_comment(self, num): 
        try:
            return self.browser.find_elements_by_xpath("//div[@class='C4VMK']/span[@class='']")[num].text
        except:
            return None
    
    def get_tags(self, separator = ' '):
        comment1, comment2 = self.get_comment(0), self.get_comment(1)
        comments = comment1 if comment2 is None else comment1+comment2
        tags = re.findall('#[A-Za-z]+ ', comments)
        return separator.join(tags)
            
        
        

    """
    Extract groups of information found in the same location and return a dict
    """
    
    
    def collect_user_metrics(self, username = None):
        #self.renavigation_check(r'https://www.instagram.com/'+ username)
        if username is not None: 
            self.browser.get(r'https://www.instagram.com/'+ username)
        time.sleep(0.5)
        user_elements = self.browser.find_elements_by_class_name('g47SY')
        num_posts = user_elements[0].text.replace(',', '')
        followers = user_elements[1].get_attribute('title').replace(',', '')
        following = user_elements[2].text.replace(',', '')
        return {'num_posts': num_posts,
               'followers': followers,
               'following': following}
    
    
    def collect_post_metrics(self):
        post_info = {}
        metrics = {'likes': self.get_likes,
                  'posttime': self.get_post_time, 
                  'alt-text': self.get_alt_text,
                  'caption': partial(self.get_comment, 0), 
                  'tags': self.get_tags}
        for feature, value in metrics.items():
            try: post_info[feature] = value()
            except: print("Couldn't collect", feature)
        return post_info
    
    
    
    def scrape_post(self, url = None, include_post = True, include_user = True):
        url = self.renavigation_check(url)
        
        time.sleep(0.1)
        post_info = self.collect_post_metrics() if include_post else {} 
        
        
        time.sleep(0.1)
        user_info = self.collect_user_metrics() if include_user else {}
        
    
        all_info = {**post_info, **user_info}
        return all_info
    
    
   
    """
    Download Posts
    """
    
    def get_post_id(self, url): return url.split(r'/')[-2]
        
    
    
    def download_post(self, url = None, output_dir = None):
        #Option of using the current page, or navigating to a specific url
        output_dir = Path('output') if output_dir is None else output_dir
        output_dir.mkdir(exist_ok=True)
        url = self.renavigation_check(url)
        #time.sleep(2)
            
        try:
            download_link = self.browser.find_elements_by_class_name('FFVAD')[0].get_attribute('src')
            urllib.request.urlretrieve(download_link, output_dir/(url.split(r'/')[-2] + '.jpg'))
        except:
            print('Couldnt print this one: ', url)
            return False
        return True






    """
    """
    
    def fill_in_columns(self, df):
        default = {'str':"", 'num':np.nan}
        columns = [('followers', 'num'), ('following', 'num'),('num_posts', 'num'), ('username', 'str'),
                   ('likes','num'), ('posttime', 'str'), ('alt-text', 'str'), ('caption', 'str'), ('tags', 'str'), 
                   ('Engagement (Avg Likes)', 'num'), ('Downloaded', 'num')]
        for (col, type_of_col) in columns:
            if col not in df.columns: df[col] = default[type_of_col]
        return df

    
    
    def fill_in_dataframe(self, df, starting_position = 0, max_posts = None, include_post = True, include_user = True, output_folder = Path("")):
        output_folder.mkdir(exist_ok = True, parents = True)
        self.current_dataframe = df
        df = self.fill_in_columns(df)
        for (index, row) in df.iloc[starting_position:,:].iterrows(): 
            print("Current getting index ", index, "Username", df.at[index, 'username'])
            url = df.loc[index,'Links']
            if np.isnan(df.at[index, 'likes']):
                #self.browser.get(url)
                url = self.renavigation_check(url)
                if not self.browser.find_elements_by_class_name('error-container'):
                    for (key, value) in self.scrape_post(include_post = include_post, include_user = include_user).items():

                            df.at[index, key] = value

                    #time.sleep(1)
                    self.download_post(url)


                    if index%20 == 0: df.to_csv(output_folder/'temp_output.csv')
                #except Exception as e: print(f'Exception: {e} at {url}')


                if max_posts: 
                    if index>max_posts: break
            else:        print('Skipping this post')
        df.to_csv(output_folder/'filled_dataframe.csv')    
        return df
    
    
    def export_dataframe(self, output_name = 'output.csv'):
        self.current_dataframe.to_csv(output_name)
        
        
        
    def load_dataframe(self, file_name = 'output.csv'):
        return pd.read_csv(file_name)
    
    
    
    def list_of_posts_from_user(self, max_posts = 1000):
        actionChain = webdriver.ActionChains(self.browser)
        all_links = set()
        old_len = -1
        while old_len!=len(all_links) and old_len<max_posts:
            old_len = len(all_links)
            actionChain.key_down(Keys.END).key_up(Keys.END).perform()
            time.sleep(2)
            links = self.browser.find_elements_by_xpath('//div[@class="v1Nh3 kIKUG  _bz0w"]/a')
            for element in links:
                try:
                    href = element.get_attribute('href')
                    if r'/p/' in href and href not in all_links: all_links.add(href)
                    #print('Got link')
                except Exception as e:
                    print('Exception', e)
                    
        user_post_list = list(all_links)
        return user_post_list
    
    def gather_posts_from_user(self, username, max_posts = 1000):
        """
        Note: this approach find all posts first, then scrapes through them
        An alternate approach involves going to the first post, then gradually iterating through them
        
        """
        self.browser.get(r'https://www.instagram.com/'+ username)
        user_info = self.collect_user_metrics()
        
        post_list = self.list_of_posts_from_user()
        new_df = pd.DataFrame({'Links':post_list, 
                               'filename': [self.get_post_id(url) for url in post_list]})
        for key, value in user_info.items():
            #new_df.assign(key=value)
            new_df[key] = value
            
        new_df['username'] = username
        
        return new_df
    
    def concat_dataframes(self, old, new): return pd.concat([old, new], ignore_index = True, sort = True)
    
    def add_links_to_dataframe_from_user(self, df, username, max_posts = 1000):
        new_df = self.gather_posts_from_user(username, max_posts)
        return self.concat_dataframes(df, new_df)
    
    
    def posts_from_list_of_users(self, user_list, max_posts = 1000, df = None):
        df = pd.DataFrame() if df is None else df
        for user in user_list:
            df = self.add_links_to_dataframe_from_user(df, user, max_posts)
            df.to_csv('temp_user_list.csv')
            
        return df
    
    
    def extract_user_list_from_dataframe(self, df, filters = []):
        mod_df = df
        for (parameter, minimum) in filters:
            mod_df = mod_df[mod_df[parameter]>minimum]
            
        return list(mod_df['username'])
    
    def download_missing(self, df, output_folder = 'Output'):
        if 'Downloaded' not in df.columns: df['Downloaded'] = np.nan
        for (index, row) in df.iterrows():
            post_url = df.at[index, 'Links']
            image_name = self.get_post_id(post_url)
            
            output_folder = Path(output_folder)
            print((output_folder/(image_name+'.jpg')))
            if not np.isnan(df.at[index, 'Downloaded']):
                print('Already processed')
            
            elif (output_folder/(image_name+'.jpg')).exists(): 
                df.at[index, 'Downloaded'] = 1
            else: 
                df.at[index, 'Downloaded'] = 1 if self.download_post(post_url) else 0

                    
            if index%20 == 0: df.to_csv('temp_output.csv')
                
        return df  
    
    
    
if __name__ == '__main__':
    print('Testing not implemented just yet... stay tuned')