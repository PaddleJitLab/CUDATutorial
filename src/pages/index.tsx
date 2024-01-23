import React, { useEffect } from 'react';
import { useHistory } from 'react-router-dom';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';
import HomepageFeatures from '../components/HomepageFeatures';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
      <header className={clsx('hero', styles.heroBanner)}>
          <div className="container">
              <img src="img/book.png" alt="" />
              <h1 className="hero__title">{siteConfig.title}</h1>
              <p className="hero__subtitle">{siteConfig.tagline}</p>
              <div className={styles.buttons}>
                  <Link
                      className="button button--secondary button--lg"
                      to="/docs/build_dev_env">
                      ENTER
                  </Link>
              </div>
          </div>
      </header>
  );
}

export default function Home(): JSX.Element {
    const { siteConfig } = useDocusaurusContext();
    const history = useHistory();

    useEffect(() => {
        // 在页面加载时进行跳转
        history.push('/docs/build_dev_env');
    }, []); // 注意空数组表示仅在组件加载时执行一次

    return (
        <Layout
            title={`Hello from ${siteConfig.title}`}
            description="Description will go into a meta tag in <head />">
            <HomepageHeader />
            <main>
                <HomepageFeatures />
            </main>
        </Layout>
    );
}
